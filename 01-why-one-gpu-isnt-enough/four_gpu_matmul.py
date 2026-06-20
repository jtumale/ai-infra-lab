"""Four-GPU tiled matmul.

Companion code for "Why One GPU Isn't Enough". Runs a 65,536 x 65,536 matmul
that OOMs on one (and even two) 32 GB cards, by splitting the output into a
2x2 grid across four GPUs.
"""
import time
import torch

GIB = 1024 ** 3
n = 65536          # OOMs on one -- and even on two -- 32 GB cards
h = n // 2         # 32768
tiles = [(0, 0), (0, 1), (1, 0), (1, 1)]   # tile -> gpu 0,1,2,3


def verify_correctness(small_n=512):
    """Run the same 2x2 tiling at a small size and compare to a single-device
    reference X @ Y. The full-size run can't do this (the reference would OOM)."""
    hs = small_n // 2
    torch.manual_seed(0)
    X = torch.randn(small_n, small_n, device="cuda:0")
    Y = torch.randn(small_n, small_n, device="cuda:0")
    ref = X @ Y

    xr_halves = [X[:hs], X[hs:]]
    yc_halves = [Y[:, :hs], Y[:, hs:]]
    Z = torch.empty(small_n, small_n, device="cuda:0")
    for gpu, (r, c) in enumerate(tiles):
        dev = f"cuda:{gpu}"
        Z[r * hs:(r + 1) * hs, c * hs:(c + 1) * hs] = (
            xr_halves[r].to(dev) @ yc_halves[c].to(dev)
        ).to("cuda:0")

    max_err = (Z - ref).abs().max().item()
    ok = torch.allclose(Z, ref, rtol=1e-3, atol=1e-2)
    print(f"correctness check (n={small_n}): max abs error = {max_err:.2e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, "Tiling does not reconstruct the reference product!"


def main():
    if torch.cuda.device_count() < 4:
        raise RuntimeError(
            f"This demo needs four GPUs; found {torch.cuda.device_count()}."
        )

    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    for g in range(4):
        print(f"cuda:{g}       {torch.cuda.get_device_name(g)}")
    print("-" * 52)

    verify_correctness()
    print("-" * 52)

    # Initialize each GPU before touching its memory counters, then reset them.
    for g in range(4):
        torch.zeros(1, device=f"cuda:{g}")
        torch.cuda.reset_peak_memory_stats(g)

    # Each GPU holds one X row-half (8 GiB) + one Y col-half (8 GiB) + its tile (4 GiB).
    x_row = [torch.randn(h, n, device="cuda:0"),   # X top rows,    home cuda:0
             torch.randn(h, n, device="cuda:2")]   # X bottom rows, home cuda:2
    y_col = [torch.randn(n, h, device="cuda:0"),   # Y left cols,   home cuda:0
             torch.randn(n, h, device="cuda:1")]   # Y right cols,  home cuda:1

    z = [None] * 4
    for g in range(4):                  # barrier: shard creation done before timing
        torch.cuda.synchronize(g)
    start = time.perf_counter()
    for gpu, (r, c) in enumerate(tiles):
        dev = f"cuda:{gpu}"
        xr = x_row[r].to(dev, non_blocking=True)   # bring this GPU's X rows here
        yc = y_col[c].to(dev, non_blocking=True)   # bring this GPU's Y cols here
        z[gpu] = xr @ yc                           # local 32768 x 32768 output tile
    for g in range(4):
        torch.cuda.synchronize(g)
    elapsed = time.perf_counter() - start

    for gpu, (r, c) in enumerate(tiles):
        peak = torch.cuda.max_memory_allocated(gpu) / GIB
        print(f"cuda:{gpu} -> tile Z[{r},{c}] | {peak:5.2f} GiB peak")
    print(f"\ncompleted {n}x{n} matmul as 4 tiles across 4 GPUs in {elapsed:.2f} s")


if __name__ == "__main__":
    main()

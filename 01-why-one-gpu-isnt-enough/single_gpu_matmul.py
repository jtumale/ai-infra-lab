"""Single-GPU matmul capacity benchmark.

Companion code for "Why One GPU Isn't Enough". Runs X @ Y at increasing square
sizes, reporting peak memory per size, until an allocation no longer fits and
PyTorch raises OutOfMemoryError.
"""
import gc
import time
import torch

device = "cuda"
GIB = 1024 ** 3
sizes = [4096, 8192, 12288, 16384, 32768, 49152, 65536]


def reset_gpu_memory():
    """Drop the prior run's cached blocks and reset the peak counter so each
    size is measured independently."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def time_matmul(size):
    """Allocate two size x size float32 inputs and time X @ Y.
    x, y, z are freed when this returns or raises."""
    x = torch.randn(size, size, device=device)  # float32 -> 4 bytes/value
    y = torch.randn(size, size, device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    z = x @ y
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required.")

    _, total_bytes = torch.cuda.mem_get_info()
    print(f"GPU:          {torch.cuda.get_device_name()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Capacity:     {total_bytes / GIB:.2f} GiB total (CUDA-reported)")
    print("-" * 52)

    for size in sizes:
        reset_gpu_memory()
        try:
            elapsed = time_matmul(size)
            peak = torch.cuda.max_memory_allocated() / GIB
            print(f"{size:>6} | {elapsed:7.4f} s | {peak:6.2f} GiB peak")
        except torch.cuda.OutOfMemoryError:
            print(f"{size:>6} | OUT OF MEMORY (failed allocating an input matrix)")
            break


if __name__ == "__main__":
    main()

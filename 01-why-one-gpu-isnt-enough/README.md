# Why One GPU Isn't Enough

Companion code for the article:
[Why One GPU Isn't Enough](https://medium.com/@justintumale/why-one-gpu-isnt-enough-what-happens-when-the-model-no-longer-fits-c8724ce7562b)

## What's here
- `single_gpu_matmul.py` — runs X @ Y at increasing sizes on one GPU until it OOMs.
- `four_gpu_matmul.py` — runs the 65,536 x 65,536 matmul (which OOMs on one and
  two 32 GB cards) by tiling the output 2x2 across four GPUs.

## Hardware
4x NVIDIA GeForce RTX 5090 (rented on vast.ai)

## Environment
PyTorch 2.12.0+cu130, CUDA runtime 13.0

## Run
```bash
pip install -r requirements.txt
python single_gpu_matmul.py
python four_gpu_matmul.py   # needs 4 GPUs
```

## Expected output
- Single GPU: sizes through 49,152 complete (~27 GiB peak); 65,536 OOMs.
- Four GPUs: correctness check PASS, peaks [28, 20, 20, 20] GiB, ~9.4 s.

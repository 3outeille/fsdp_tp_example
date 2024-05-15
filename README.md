# NCCL backend

## Float64

`torchrun --nproc_per_node=4 test_fsdp_tp_dmesh.py --tp=2 --dp=2 --no_wandb --dtype=float64` -> ✅
`python test_fsdp_tp.py --tp=2 --dp=2 --no_wandb --dtype=float64` ->  ✅

## Float32

`torchrun --nproc_per_node=4 test_fsdp_tp_dmesh.py --tp=2 --dp=2 --no_wandb --dtype=float32` -> ❌
`python test_fsdp_tp.py --tp=2 --dp=2 --no_wandb --dtype=float32` -> ❌

- The issue seems to come from `TP` alone and `float32` precision
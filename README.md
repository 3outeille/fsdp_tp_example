Minimum working example of fsdp and tp (using dmesh and process groups)

```
# Use TP with dmesh
python test_fsdp_tp.py --tp=2 --no_wandb
# Use TP with megatron style
torchrun --nproc_per_node=2 test_fsdp_tp_dmesh.py --tp=2 --no_wandb
```
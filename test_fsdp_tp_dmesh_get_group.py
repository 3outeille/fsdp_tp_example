"""
torchrun --nproc_per_node=2 test_fsdp_tp_dmesh_get_group.py --tp=2 --no_wandb --dtype=float64
torchrun --nproc_per_node=4 test_fsdp_tp_dmesh_get_group.py --tp=2 --dp=2 --no_wandb --dtype=float64
"""
import os
import functools
import torch
import argparse
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import wandb
import datetime
from copy import deepcopy
import os
import shutil
from utils import (
    MNISTloader,
    set_random_seed,
    split_weight,
    print_rank_0,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    loss_parallel,
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from torch.distributed._tensor import Shard, Replicate
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy
)
import lovely_tensors as lt; lt.monkey_patch()
import torch.distributed.checkpoint as dcp

class DummyModel(nn.Module):
    def __init__(self, input_size, output_size, bias):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512, bias=bias)
        self.fc2 = nn.Linear(512, 256, bias=bias)
        self.fc3 = nn.Linear(256, 128, bias=bias)
        self.fc4 = nn.Linear(128, output_size, bias=bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def train(
    seed: int,
    lr: int,
    epochs: int,
    batch_size: int,
    no_wandb: bool,
    dtype: torch.dtype,
    dp: int = 1,
    tp: int = 1,
):
    set_random_seed(seed)

    device_mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))

    train_loader, _, _ = MNISTloader(batch_size=batch_size, train_val_split=0.5).load()
    
    dist.barrier()
    
    ref_model = DummyModel(
        input_size=32 * 32,
        output_size=10,
        bias=False
    )
    
    model = deepcopy(ref_model)
    
    model.to("cuda", dtype=dtype)
    device = next(model.parameters()).device
    ref_model.to(device, dtype=dtype)
    
    # assert if parameters are the same
    for model_param, ref_model_param in zip(model.parameters(), ref_model.parameters()):
        model_param.data.copy_(ref_model_param.data)
        torch.testing.assert_close(model_param, ref_model_param, rtol=1e-3, atol=1e-3)  
    
    dist.barrier()
    
    # Apply ColumnParallel to fc1 then RowParallel to fc2
    if tp > 1:
        model = parallelize_module(
            model,
            device_mesh["tp"],
            {
                "fc1": ColwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "fc2": RowwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate(),
                ),
                "fc3": ColwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "fc4": RowwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate(),
                )
            }
        )
    
        
    # Apply FSDP
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1 # Will create 1 unit for each layer
    )
    
    model = FSDP(model, process_group=device_mesh.get_group("dp"), auto_wrap_policy=my_auto_wrap_policy, use_orig_params=True)
        
    optim = Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    ref_optim = Adam(ref_model.parameters(), lr)
    ref_criterion = nn.CrossEntropyLoss()
    
    model.train()
    ref_model.train()
    dist.barrier()
    
    if dist.get_rank() == 0 and not no_wandb:
        def get_time_name():
            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")
        
        wandb.init(
            project="fsdp-tp",
            name=f"test_tp_mnist_convergence_{get_time_name()}",
            config={
                "model": "NN",
                "dataset": "MNIST",
                "epochs": epochs,
                "learning_rate": lr,
                "seed": seed,
            },
        )
    
    for epoch in range(epochs):
        for step, (inputs, target) in enumerate(train_loader):
            inputs, target = inputs.to("cuda", dtype), target.to("cuda")
            
            inputs = inputs.flatten(1)
        
            ref_logits = ref_model(inputs)
            ref_loss = ref_criterion(ref_logits, target)

            ref_optim.zero_grad()
            ref_loss.backward()
            ref_optim.step()

            with loss_parallel():
                logits = model(inputs)
                loss = criterion(logits, target)
                    
                optim.zero_grad()
                loss.backward()
                optim.step()    
            
            torch.testing.assert_close(logits, ref_logits, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(loss.item(), ref_loss.item(), rtol=1e-3, atol=1e-3)
            
            if step % 1000 == 0:
                print_rank_0(f"Epoch: {epoch} | Step: {step}, Loss: {loss.item()}, Ref Loss: {ref_loss.item()}")

            if dist.get_rank() == 0 and not no_wandb:
                wandb.log(
                    {
                        "train_loss": loss,
                        "ref_train_loss": ref_loss,
                        "epoch": epoch,
                        "step": step,
                    }
                )
    
    # save model
    print_rank_0("Saving model")
    
    if dist.get_rank() == 0:
        if os.path.exists("model_ckpt_dmesh"):
            shutil.rmtree("model_ckpt_dmesh")
        os.makedirs("model_ckpt_dmesh")
        if os.path.exists("ref_model_dmesh.pth"):
            os.remove("ref_model_dmesh.pth")
    
    dist.barrier()
    
    torch.save(ref_model.state_dict(), "ref_model_dmesh.pth")
    if tp > 1:
        dcp.save(model.state_dict(), checkpoint_id="model_ckpt_dmesh")
    else:
        torch.save(model.state_dict(), "model_ckpt_dmesh.pth")
    
    print_rank_0("Reload model")
    
    del model
    del ref_model
    
    model = None
    ref_model = None
    
    ref_model = DummyModel(
        input_size=32 * 32,
        output_size=10,
        bias=False
    )
    
    model = deepcopy(ref_model)
    
    if tp > 1:
        model = parallelize_module(
            model,
            device_mesh["tp"],
            {
                "fc1": ColwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "fc2": RowwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate(),
                ),
                "fc3": ColwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "fc4": RowwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate(),
                )
            }
        )
    
    model.to("cuda", dtype)
    ref_model.to("cuda", dtype)
        
    ref_model.load_state_dict(torch.load("ref_model_dmesh.pth"))
    if tp > 1:
        dcp.load(model.state_dict(), checkpoint_id="model_ckpt_dmesh")
    else:
        model.load_state_dict(torch.load("model_ckpt_dmesh.pth"))
        
    dist.barrier()
    
    # Recreate optimizer
    optim = Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    ref_optim = Adam(ref_model.parameters(), lr)
    ref_criterion = nn.CrossEntropyLoss()
    
    model.train()
    ref_model.train()
    dist.barrier()
    
    print_rank_0("Training again after reloading model")
    
    for epoch in range(epochs):
        for step, (inputs, target) in enumerate(train_loader):
            inputs, target = inputs.to("cuda", dtype), target.to("cuda")
            
            inputs = inputs.flatten(1)
        
            ref_logits = ref_model(inputs)
            ref_loss = ref_criterion(ref_logits, target)

            ref_optim.zero_grad()
            ref_loss.backward()
            ref_optim.step()

            with loss_parallel():
                logits = model(inputs)
                loss = criterion(logits, target)
                    
                optim.zero_grad()
                loss.backward()
                optim.step()    
            
            torch.testing.assert_close(logits, ref_logits, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(loss.item(), ref_loss.item(), rtol=1e-3, atol=1e-3)
            
            if step % 1000 == 0:
                print_rank_0(f"Epoch: {epoch} | Step: {step}, Loss: {loss.item()} Ref Loss: {ref_loss.item()}")

            if dist.get_rank() == 0 and not no_wandb:
                wandb.log(
                    {
                        "train_loss": loss,
                        "ref_train_loss": ref_loss,
                        "epoch": epoch,
                        "step": step,
                    }
                )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--dtype", type=str, default="float64")

    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    SEED = 42
    LR = 3e-4
    BATCH_SIZE = 6
    EPOCHS = 1
    
    dtype_lut = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    
    train(
        seed=SEED,
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        no_wandb=args.no_wandb,
        dtype=dtype_lut[args.dtype],
        dp=args.dp,
        tp=args.tp,
    )
    

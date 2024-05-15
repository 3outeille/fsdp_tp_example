"""
python test_fsdp_tp.py --tp=2 --no_wandb --dtype=float64
python test_fsdp_tp.py --tp=2 --dp=2 --no_wandb --dtype=float64
"""
import functools
import os
import shutil
import torch
import argparse
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import wandb
import datetime
from utils import (
    init_distributed,
    MNISTloader,
    set_random_seed,
    split_weight,
    print_rank_0,
)
from tensor_parallel import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    TensorParallelLinearMode,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import loss_parallel
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy
)
import lovely_tensors as lt; lt.monkey_patch()


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

class TensorParallelDummyModel(nn.Module):
    def __init__(self, input_size, output_size, parallel_context, tp_mode, async_communication, bias):
        super(TensorParallelDummyModel, self).__init__()
        self.fc1 = TensorParallelColumnLinear(
            in_features=input_size,
            out_features=512,
            pg=parallel_context.tp_pg,
            mode=tp_mode,
            device="cuda",
            async_communication=async_communication,
            bias=bias,
        )
        
        self.fc2 = TensorParallelRowLinear(
            in_features=512,
            out_features=256,
            pg=parallel_context.tp_pg,
            mode=tp_mode,
            device="cuda",
            async_communication=async_communication,
            bias=bias,
        )
        
        self.fc3 = TensorParallelColumnLinear(
            in_features=256,
            out_features=128,
            pg=parallel_context.tp_pg,
            mode=tp_mode,
            device="cuda",
            async_communication=async_communication,
            bias=bias,
        )
        
        self.fc4 = TensorParallelRowLinear(
            in_features=128,
            out_features=output_size,
            pg=parallel_context.tp_pg,
            mode=tp_mode,
            device="cuda",
            async_communication=async_communication,
            bias=bias,
        )


    def forward(self, x):
        x = F.relu(self.fc1(x)) # fc1.weight: (512, 1024) -> (256, 1024)
        x = F.relu(self.fc2(x)) # fc2.weight: (10, 512) -> (10, 256)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)        
        return x

    
def train(
    parallel_context,
    seed: int,
    lr: int,
    epochs: int,
    no_wandb: bool,
    dtype: torch.dtype,
    tp_mode: TensorParallelLinearMode,
    async_communication: bool
):
    set_random_seed(seed)

    train_loader, _, _ = MNISTloader(batch_size=6, train_val_split=0.5).load()
    
    dist.barrier()
    
    model = TensorParallelDummyModel(
        input_size=32 * 32,
        output_size=10,
        parallel_context=parallel_context,
        tp_mode=tp_mode,
        async_communication=async_communication,
        bias=False
    )
    ref_model = DummyModel(
        input_size=32 * 32,
        output_size=10,
        bias=False
    )
    
    model.to("cuda", dtype=dtype)
    device = next(model.parameters()).device
    ref_model.to(device, dtype=dtype)
    
    # Copy weights/bias from sharded to un-sharded
    with torch.inference_mode():
        
        for ref_layer, layer in zip(ref_model.children(), model.children()):    
            dim = 0 if isinstance(layer, TensorParallelColumnLinear) else 1

            dist.all_gather(
                tensor_list=list(ref_layer.weight.split(ref_layer.weight.shape[dim] // parallel_context.tp_pg.size(), dim=dim)),
                tensor=layer.weight,
                group=parallel_context.tp_pg,
            )
            #TODO: bias
             
            dist.barrier()
            torch.testing.assert_close(layer.weight, split_weight(ref_layer.weight, dim=dim), rtol=1e-3, atol=1e-3)
            dist.barrier()

    dist.barrier()
    
    # Apply FSDP
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1 # Will create 1 unit for each layer
    )
    
    if parallel_context.dp_pg.size() > 1:
        model = FSDP(model, process_group=parallel_context.dp_pg, auto_wrap_policy=my_auto_wrap_policy, use_orig_params=True)
    
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
                print_rank_0(f"Epoch: {epoch} | Step: {step}, Loss: {loss.item()} ref_loss: {ref_loss.item()}")

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
        if os.path.exists("model_ckpt"):
            shutil.rmtree("model_ckpt")
        os.makedirs("model_ckpt")
        if os.path.exists("ref_model.pth"):
            os.remove("ref_model.pth")
        
    dist.barrier()
    torch.save(ref_model.state_dict(), "ref_model.pth")
    dcp.save(model.state_dict(), checkpoint_id="model_ckpt")
     
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
    
    model = TensorParallelDummyModel(
        input_size=32 * 32,
        output_size=10,
        parallel_context=parallel_context,
        tp_mode=tp_mode,
        async_communication=async_communication,
        bias=False
    )
    
    model.to("cuda", dtype=dtype)
    ref_model.to("cuda", dtype=dtype)
    
    ref_model.load_state_dict(torch.load("ref_model.pth"))
    dcp.load(model.state_dict(), checkpoint_id="model_ckpt")
        
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
                print_rank_0(f"Epoch: {epoch} | Step: {step}, Loss: {loss.item()}, ref_loss: {ref_loss.item()}")

            if dist.get_rank() == 0 and not no_wandb:
                wandb.log(
                    {
                        "train_loss": loss,
                        "ref_train_loss": ref_loss,
                        "epoch": epoch,
                        "step": step,
                    }
                )
    
    
    parallel_context.destroy()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--dtype", type=str, default="float64")
    
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    SEED = 42
    LR = 3e-4
    EPOCHS = 1
    
    dtype_lut = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    
    init_distributed(tp=args.tp, dp=args.dp, pp=args.pp)(train)(
        seed=SEED,
        lr=LR,
        epochs=EPOCHS,
        no_wandb=args.no_wandb,
        dtype=dtype_lut[args.dtype],
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        async_communication=False
    )
from typing import Optional

import torch
from torch import distributed as dist
from torch.distributed import ProcessGroup

class DifferentiableIdentity(torch.autograd.Function):
    """All-reduce gradients in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllReduceSum.apply(grad_output, group), None


class DifferentiableAllReduceSum(torch.autograd.Function):
    """All-reduce in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        if group.size() == 1:
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)        
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DifferentiableAllGather(torch.autograd.Function):
    """All gather in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        if group.size() == 1:
            return tensor

        sharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = dist.distributed_c10d._get_default_group()
        unsharded_batch_size = sharded_batch_size * group.size()

        unsharded_tensor = torch.empty(
            unsharded_batch_size,
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

        # `tensor` can sometimes not be contiguous
        # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L317
        tensor = tensor.contiguous()

        dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group)
        return unsharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableReduceScatterSum.apply(grad_output, group), None


class DifferentiableReduceScatterSum(torch.autograd.Function):
    """Reduce scatter in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        if group.size() == 1:
            return tensor

        unsharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = dist.distributed_c10d._get_default_group()
        assert unsharded_batch_size % group.size() == 0

        # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L305
        tensor = tensor.contiguous()

        sharded_tensor = torch.empty(
            unsharded_batch_size // group.size(),
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )
        dist.reduce_scatter_tensor(sharded_tensor, tensor, group=group, op=dist.ReduceOp.SUM)
        return sharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllGather.apply(grad_output, group), None


# -----------------
# Helper functions.
# -----------------


def differentiable_identity(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableIdentity.apply(tensor, group)


def differentiable_all_reduce_sum(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableAllReduceSum.apply(tensor, group)


def differentiable_all_gather(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableAllGather.apply(tensor, group)


def differentiable_reduce_scatter_sum(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableReduceScatterSum.apply(tensor, group)
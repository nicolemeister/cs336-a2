import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy


class NaiveDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super(NaiveDDP, self).__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized")

        self.module = module

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Initially, each device constructs a (randomly-initialized) model. We use the broadcast collective communication operation to send the model
        # parameters from rank 0 to all other ranks. At the start of training, each device holds an identical copy of
        # the model parameters and optimizer states (e.g. the accumulated gradient statistics in Adam).
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
    
    def forward(self, *inputs, **kwargs):
        
        # Each device uses its local copy of the model parameters to run a forward pass on its n/d examples and
        # a backward pass to calculate the gradients. Note that at this point, each device holds the gradients
        # computed from the n/d examples it received.
        y = self.module(*inputs, **kwargs)
        return y

    def finish_gradient_synchronization(self):

        # We then use the all-reduce collective communication operation to average the gradients across the
        # different devices, so each device holds the gradients averaged across all n examples.
        for param in self.module.parameters():
            if param.requires_grad:
                # divide gradient by number of gpus = world_size
                param.grad.data /= self.world_size
                # sum the gradients across all gpus (size we avgs over gpus before, this will be the average of gradients, not sum)
                dist.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM, async_op=False)
                # TODO: could try avg like in lecture: dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super(DDPIndividualParameters, self).__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized")

        self.module = module

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.handles = []

        def register_post_accumulate_grad_hook(param):
            grad = param.grad
            if grad is not None:
                grad /= self.world_size
                handle = dist.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, async_op=True)
                self.handles.append(handle)
                

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(register_post_accumulate_grad_hook)
            dist.broadcast(param.data, src=0)
    
    def forward(self, *inputs, **kwargs):
        y = self.module(*inputs, **kwargs)
        return y

    def finish_gradient_synchronization(self):

        for handle in self.handles:
            handle.wait()
        self.handles.clear()

class BucketedDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float) -> None:
        super(BucketedDDP, self).__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized")

        self.module = module

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.bucket_size_mb = bucket_size_mb

        self.handles = []
        self.buckets = []
        

        def register_post_accumulate_grad_hook(param, bucket_index):
            self.bucket_ready[bucket_index] += 1
            if self.bucket_ready[bucket_index] == len(self.buckets[bucket_index]):
                grads = [param.grad for param in self.buckets[bucket_index]]
                if grads:
                    dense = _flatten_dense_tensors(grads)
                    handle = dist.all_reduce(dense, op=torch.distributed.ReduceOp.SUM, async_op=True)
                    self.handles.append((handle, dense, bucket_index))
                self.bucket_ready[bucket_index] = 0
                

        current_bucket = []
        current_bucket_size = 0
        
        mb_to_bytes = 1024**2
        max_bucket_bytes = self.bucket_size_mb * mb_to_bytes

        for param in reversed(list(self.module.parameters())):
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param_size = param.data.numel() * param.data.element_size()
                if current_bucket_size + param_size > max_bucket_bytes:
                    self.buckets.append(current_bucket)
                    current_bucket = []
                    current_bucket_size = 0
                current_bucket.append(param)
                current_bucket_size += param_size
        
        if current_bucket:
            self.buckets.append(current_bucket)

        self.bucket_ready = [0 for _ in range(len(self.buckets))]

        for i, bucket in enumerate(self.buckets):
            for param in bucket:
                param.register_post_accumulate_grad_hook(lambda param, bucket_index=i: register_post_accumulate_grad_hook(param, bucket_index))
    
    def forward(self, *inputs, **kwargs):
        y = self.module(*inputs, **kwargs)
        return y

    def finish_gradient_synchronization(self):

        for handle, dense, bucket_index in self.handles:
            grads = [param.grad for param in self.buckets[bucket_index] if param.grad is not None]
            handle.wait()
            unflattened_tensors = _unflatten_dense_tensors(dense, grads)
            for param, unflattened in zip(self.buckets[bucket_index], unflattened_tensors):
                param.grad.data = unflattened / self.world_size
        self.handles.clear()
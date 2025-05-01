'''
Benchmark your attention implementation at different scales. Write a script that will:
(c) Create random inputs Q, K, V for the appropriate size.
(e) Measure how much memory is in use before the backward pass starts, and time 100 backward
passes.


'''

import torch
import sys
import triton
import time
import os
import cs336_basics.data as cs336_data
import cs336_basics.model as cs336_model
import cs336_basics.optimizer as cs336_optim
import cs336_basics.nn_utils as cs336_utils
from cs336_systems.flash_attention_pytorch import FlashAttention2Func
from cs336_systems.flash_attention_triton import TritonFlashAttention
import timeit
import einops
import pandas as pd
import numpy as np
import argparse

'''
 Write a benchmarking script using triton.testing.do_bench that compares the performance
of your (partially) Triton implementation of FlashAttention-2 forward and backward passes with
a regular PyTorch implementation (i.e., not using FlashAttention).
Specifically, you will report a table that includes latencies for forward, backward, and the endto-end forward-backward pass, for both your Triton and PyTorch implementations. Randomly
generate any necessary inputs before you start benchmarking, and run the benchmark on a single
H100. Always use batch size 1 and causal masking. Sweep over the cartesian product of sequence
lengths of various powers of 2 from 128 up to 65536, embedding dimension sizes of various powers
of 2 from 16 up to size 128, and precisions of torch.bfloat16 and torch.float32. You will
likely need to adjust tile sizes depending on the input sizes.
'''

# argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--compile_attention", type=int, default=0)
args = parser.parse_args()

compile_attention = args.compile_attention


d_ff = 3072
num_layers = 12
num_heads = 1
num_warmups = 5
num_trials = 10
memory_profile = 0

# FIXED HYPERPARAMETERS
vocab_size = 10000
batch_size = 8
num_heads=1
rope_theta = 10000.0
device = "cuda" if torch.cuda.is_available() else "cpu"


for context_length in [256, 512, 1024, 2048, 4096, 8192, 16384]:
    for d_model in [16, 32, 64, 128]: 
        # generate random batch of data
        min_dataset_size = 2 * context_length
        dataset = torch.randint(0, vocab_size, (max(vocab_size, min_dataset_size),)).numpy()

        data_x, data_y = cs336_data.get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device
        )

        print("Using device: ", device)
    
        n_queries = 128
        n_keys = 128
        D = 64

        Q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
        K = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
        V = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
        L = torch.randn(batch_size, context_length, device=device)  # L has no d_model dimension
        dO = torch.randn(batch_size, n_queries, D, device=device)


        def forward_fn():
            if compile_attention:
                compiled_attention = torch.compile(cs336_model.scaled_dot_product_attention)
                output = compiled_attention(Q, K, V, None)
            else:
                output = cs336_model.scaled_dot_product_attention(Q, K, V, None)
            return output

        def backward_fn(output):
            if compile_attention:
                compiled_backward = torch.compile(output.backward)
                compiled_backward(dO)
            else:
                output.backward(dO)

        def forward_backward_fn():
            output = forward_fn()
            output.backward(dO)


        # Warmup
        for _ in range(num_warmups):
            torch.cuda.synchronize()
            forward_backward_fn()
            torch.cuda.synchronize()

        # Time forward pass
        forward_times = []
        backward_times = []
        end_to_end_times = []
        for _ in range(num_trials):
            torch.cuda.synchronize()
            
            start_time_forward = timeit.default_timer()
            dO = forward_fn()
            torch.cuda.synchronize()
            end_time_forward = timeit.default_timer()
            
            forward_times.append((end_time_forward - start_time_forward) * 1000)

            torch.cuda.synchronize()
            start_time_backward = timeit.default_timer()
            backward_fn(dO)
            torch.cuda.synchronize()
            end_time_backward = timeit.default_timer()
            backward_times.append((end_time_backward - start_time_backward) * 1000)
            
            end_to_end_times.append((end_time_backward - start_time_forward) * 1000)

        # save the mean time for forward and backward passes in a pandas dataframe with hyperparameters in the filename
        df = pd.DataFrame({
            "d_model": d_model,
            "seq_len": context_length,
            "compile_attention": compile_attention,
            "forward_times_mean": [f"{sum(forward_times) / len(forward_times):.4f} ms"],
            "backward_times_mean": [f"{sum(backward_times) / len(backward_times):.4f} ms"],
            "end_to_end_times_mean": [f"{sum(end_to_end_times) / len(end_to_end_times):.4f} ms"],
            "forward_times_std": [f"{np.std(forward_times):.4f} ms"],
            "backward_times_std": [f"{np.std(backward_times):.4f} ms"],
            "end_to_end_times_std": [f"{np.std(end_to_end_times):.4f} ms"],

        })
            
        # save forward and backward times to a file with hyperparameters in the filename
        os.makedirs("pytorch_attention_benchmarking", exist_ok=True)

        # save the dataframe to a csv file
        df.to_csv(f"pytorch_attention_benchmarking/dmodel_{d_model}_context_length_{context_length}_compile_attention_{compile_attention}.csv", index=False)

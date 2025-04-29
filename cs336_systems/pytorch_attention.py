'''
Benchmark your attention implementation at different scales. Write a script that will:
(c) Create random inputs Q, K, V for the appropriate size.
(e) Measure how much memory is in use before the backward pass starts, and time 100 backward
passes.


'''


import torch
import sys
import time
import os
import cs336_basics.data as cs336_data
import cs336_basics.model as cs336_model
import cs336_basics.optimizer as cs336_optim
import cs336_basics.nn_utils as cs336_utils
import timeit
import einops
import pandas as pd
import numpy as np


# take in command line arguments: d_model d_ff num_layers num_heads
import argparse

parser = argparse.ArgumentParser(description='Benchmark a Transformer model')
parser.add_argument('--name', type=str, required = True, help='Model name')
parser.add_argument('--d_model', type=int, required = True, help='Model dimension')
parser.add_argument('--d_ff', type=int, required = True, help='Feed-forward dimension')
parser.add_argument('--num_layers', type=int, required = True, help='Number of layers')
parser.add_argument('--num_heads', type=int, required=True, help='Number of attention heads')
parser.add_argument('--num_warmups', type=int, default=5, help='Number of warmup steps')
parser.add_argument('--context_length', type=int, default=128, help='Context length')
parser.add_argument('--num_trials', type=int, default=10, help='Number of trials')
parser.add_argument('--use_mixed_precision', type=int, default=0, help='Use mixed precision (0/1)')
parser.add_argument('--memory_profile', type=int, default=0, help='Enable memory profiling (0/1)')
parser.add_argument('--compile_model', type=int, default=0, help='Compile model (0/1)')
parser.add_argument('--compile_attention', type=int, default=0, help='Compile model (0/1)')

args = parser.parse_args()

name = args.name
d_model = args.d_model
d_ff = args.d_ff
num_layers = args.num_layers
num_heads = args.num_heads
num_warmups = args.num_warmups
context_length = args.context_length
num_trials = args.num_trials
use_mixed_precision = args.use_mixed_precision
compile_model = args.compile_model
compile_attention = args.compile_attention
memory_profile = args.memory_profile


'''
# to do:

set numheads to 1


'''


# FIXED HYPERPARAMETERS
vocab_size = 10000
batch_size = 8
num_heads=1
rope_theta = 10000.0
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device: ", device)

if compile_attention:
    cs336_model.scaled_dot_product_attention = torch.compile(cs336_model.scaled_dot_product_attention)


model = cs336_model.BasicsTransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta,
).to(device)


if compile_model:
    model = torch.compile(model)


# generate random batch of data
min_dataset_size = 2 * context_length
dataset = torch.randint(0, vocab_size, (max(vocab_size, min_dataset_size),)).numpy()

data_x, data_y = cs336_data.get_batch(
    dataset=dataset,
    batch_size=batch_size,
    context_length=context_length,
    device=device
)

optimizer = cs336_optim.AdamW(model.parameters(), lr=1e-3)

# Forward pass timing
def forward_pass():
    if use_mixed_precision: 
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model.forward(data_x)
    else:   
        logits = model.forward(data_x)
    return logits

# Backward pass timing
def backward_pass(loss):
    
    torch.cuda.synchronize()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()
    return


# Warmup
for _ in range(num_warmups):
    torch.cuda.synchronize()
    logits = forward_pass()
    loss = cs336_utils.cross_entropy(
        einops.rearrange(logits, 'b c v -> (b c) v'),
        einops.rearrange(data_y, 'b c -> (b c)')
    )
    backward_pass(loss)
    torch.cuda.synchronize()

# Time forward pass
forward_times = []
backward_times = []
memory_usage = []
for _ in range(num_trials):
    torch.cuda.synchronize()
    
    if memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    start_time_forward = timeit.default_timer()
    logits = forward_pass()
    torch.cuda.synchronize()
    end_time_forward = timeit.default_timer()
    if memory_profile:
        # Save a pickle file to be loaded by PyTorch's online tool.
        torch.cuda.memory._dump_snapshot(f"memory_snapshot/forwardpass_{name}_num_warmups_{num_warmups}_context_length_{context_length}_mixed_precision_{use_mixed_precision}.pickle")
        # Stop recording history.
        torch.cuda.memory._record_memory_history(enabled=None)

    forward_times.append((end_time_forward - start_time_forward) * 1000)

    loss = cs336_utils.cross_entropy(
        einops.rearrange(logits, 'b c v -> (b c) v'),
        einops.rearrange(data_y, 'b c -> (b c)')
    )

    memory_before_backward = torch.cuda.max_memory_allocated()
    memory_usage.append(memory_before_backward)

    start_time_backward = timeit.default_timer()
    torch.cuda.synchronize()
    backward_pass(loss)
    torch.cuda.synchronize()
    end_time_backward = timeit.default_timer()
    if memory_profile:
        # Save a pickle file to be loaded by PyTorch's online tool.
        torch.cuda.memory._dump_snapshot(f"memory_snapshot/fulltrainstep_{name}_num_warmups_{num_warmups}_context_length_{context_length}_mixed_precision_{use_mixed_precision}.pickle")
        # Stop recording history.
        torch.cuda.memory._record_memory_history(enabled=None)
    backward_times.append((end_time_backward - start_time_backward) * 1000)

# save forward and backward times to a file with hyperparameters in the filename
os.makedirs("pytorch_attention", exist_ok=True)

# save the mean time for forward and backward passes in a pandas dataframe with hyperparameters in the filename
df = pd.DataFrame({
    "name": [name],
    "num_warmups": [num_warmups], 
    "forward_times_mean": [f"{sum(forward_times) / len(forward_times):.4f} ms"],
    "backward_times_mean": [f"{sum(backward_times) / len(backward_times):.4f} ms"],
    "forward_times_std": [f"{np.std(forward_times):.4f} ms"],
    "backward_times_std": [f"{np.std(backward_times):.4f} ms"],
    "memory_usage": [f"{(sum(memory_usage)/len(memory_usage)) / 1024**2:.2f} MB"], 
    "compile_model": compile_model, 
    "compile_attention": compile_attention, 
    "d_model": d_model,
    "seq_len": context_length,
})

# save the dataframe to a csv file
df.to_csv(f"pytorch_attention/{name}_dmodel_{d_model}_context_length_{context_length}_compile_model_{compile_model}_compile_attention_{compile_attention}.csv", index=False)

# save the dataframe to a latex table file
# df.to_latex(f"benchmark/{name}_num_warmups_{num_warmups}.txt", index=False)


'''

• Given hyperparameters (e.g., number of layers), initialize a model.
• Generate a random batch of data.
• Run w warm-up steps (before you start measuring time), then time the execution of n steps
(either only forward, or both forward and backward passes, depending on an argument). For
timing, you can use the Python timeit module (e.g., either using the timeit function, or
using timeit.default_timer(), which gives you the system's highest resolution clock, thus
a better default for benchmarking than time.time()).
• Call torch.cuda.synchronize() after each step.
Deliverable: A script that will initialize a basics Transformer model with the given hyperparameters, create a random batch of data, and time forward and backward passes.

'''


import torch
import sys
import os
import cs336_basics.data as cs336_data
import cs336_basics.model as cs336_model
import cs336_basics.optimizer as cs336_optim
import cs336_basics.nn_utils as cs336_utils
import einops
import pandas as pd
import numpy as np
import torch.cuda.nvtx as nvtx


# take in command line arguments: d_model d_ff num_layers num_heads
name = sys.argv[1]
d_model = int(sys.argv[2])
d_ff = int(sys.argv[3])
num_layers = int(sys.argv[4])
num_heads = int(sys.argv[5])
num_warmups = int(sys.argv[6])
context_length = int(sys.argv[7])
if len(sys.argv) > 8:
    num_trials = int(sys.argv[8])
else:
    num_trials = 10

# FIXED HYPERPARAMETERS
vocab_size = 10000
batch_size = 4
rope_theta = 10000.0
device = "cuda" if torch.cuda.is_available() else "cpu"



cs336_model.scaled_dot_product_attention = cs336_model.annotated_scaled_dot_product_attention

model = cs336_model.BasicsTransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta,
).to(device)


# generate random batch of data
dataset = torch.randint(0, vocab_size, (vocab_size,)).numpy()

data_x, data_y = cs336_data.get_batch(
    dataset=dataset,
    batch_size=batch_size,
    context_length=context_length,
    device=device
)

optimizer = cs336_optim.AdamW(model.parameters(), lr=1e-3)

# Backward pass
def backward_pass(loss):
    
    return


# Warmup
nvtx.range_push("warmup")
for _ in range(num_warmups):
    torch.cuda.synchronize()
    logits = model.forward(data_x)
    torch.cuda.synchronize()
    loss = cs336_utils.cross_entropy(
        einops.rearrange(logits, 'b c v -> (b c) v'),
        einops.rearrange(data_y, 'b c -> (b c)')
    )
    torch.cuda.synchronize()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()
    optimizer.step()
    torch.cuda.synchronize()
nvtx.range_pop()

# Run trials
nvtx.range_push("training_steps")
for _ in range(num_trials):
    
    
    torch.cuda.synchronize()

    nvtx.range_push("forward")
    logits = model.forward(data_x)
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    torch.cuda.synchronize()
    loss = cs336_utils.cross_entropy(
        einops.rearrange(logits, 'b c v -> (b c) v'),
        einops.rearrange(data_y, 'b c -> (b c)')
    )

    torch.cuda.synchronize()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    nvtx.range_push("backward")
    loss.backward()
    torch.cuda.synchronize()
    nvtx.range_pop()
    torch.cuda.synchronize()

    nvtx.range_push("optimizer_step")
    optimizer.step()
    torch.cuda.synchronize()
    nvtx.range_pop()
nvtx.range_pop()

# save results directory
os.makedirs("benchmark", exist_ok=True)

# create dataframe with model parameters
df = pd.DataFrame({
    "name": [name],
    "num_warmups": [num_warmups]
})

# save the dataframe to a csv file
df.to_csv(f"benchmark/{name}_num_warmups_{num_warmups}.csv", index=False)

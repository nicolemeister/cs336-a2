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

import torch._dynamo
torch._dynamo.config.suppress_errors = True


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


for context_length in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    for d_model in [16, 32, 64, 128]: 
    # for d_model in [64, 128]: 

        # generate random batch of data
        min_dataset_size = 2 * context_length
        dataset = torch.randint(0, vocab_size, (max(vocab_size, min_dataset_size),)).numpy()

        data_x, data_y = cs336_data.get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device
        )

        for use_mixed_precision in [0, 1]:
            for implementation in ["triton"]:

                # see if this csv exists and it is not empty
                if os.path.exists(f"flash_benchmarking/{implementation}_dmodel_{d_model}_context_length_{context_length}_use_mixed_precision_{use_mixed_precision}.csv") and os.path.getsize(f"flash_benchmarking/{implementation}_dmodel_{d_model}_context_length_{context_length}_use_mixed_precision_{use_mixed_precision}.csv") > 0:
                    print(f"Skipping {implementation} {d_model} {context_length} {use_mixed_precision} because it already exists")
                    continue


                print("Using device: ", device)
            
                n_queries = context_length
                n_keys = context_length
                D = d_model

                Q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
                K = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
                V = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
                L = torch.randn(batch_size, context_length, device=device)  # L has no d_model dimension
                dO = torch.randn(batch_size, n_queries, D, device=device)

                if implementation == "triton":
                    try: 
                        flash_attention_backward = TritonFlashAttention.flash_attention_backward

                        forward_fn = lambda: TritonFlashAttention.apply(Q, K, V, True)
                        backward_fn = lambda: flash_attention_backward(q=Q, k=K, v=V, l=L, grad_out=dO, d=D)
                        forward_backward_fn = lambda: TritonFlashAttention.apply(Q, K, V, True).backward(dO)

                        forward_time = triton.testing.do_bench(forward_fn, warmup=10000, rep=10000, return_mode='mean')
                        forward_backward_time = triton.testing.do_bench(forward_backward_fn, warmup=10000, rep=10000, return_mode='mean')
                        backward_time = triton.testing.do_bench(backward_fn, warmup=10000, rep=10000, return_mode='mean')


                        # save the mean time for forward and backward passes in a pandas dataframe with hyperparameters in the filename
                        df = pd.DataFrame({
                            "implementation": implementation,
                            "d_model": d_model,
                            "seq_len": context_length,
                            "use_mixed_precision": use_mixed_precision,
                            "forward_times_mean": [f"{forward_time:.4f} ms"],
                            "backward_times_mean": [f"{backward_time:.4f} ms"],
                            "forward_backward_times_mean": [f"{forward_backward_time:.4f} ms"],
                        })
                    except Exception as e:
                        print(f"Error: {e}")
                        df = pd.DataFrame()


                else:
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
                    end_to_end_times = []
                    for _ in range(num_trials):
                        torch.cuda.synchronize()
                        
                        
                        start_time_forward = timeit.default_timer()
                        logits = forward_pass()
                        torch.cuda.synchronize()
                        end_time_forward = timeit.default_timer()
                        
                        if implementation == "triton":
                            forward_times.append(triton.testing.do_bench(forward_fn, warmup=10000, rep=10000, return_mode='mean'))
                        else: 
                            forward_times.append((end_time_forward - start_time_forward) * 1000)

                        loss = cs336_utils.cross_entropy(
                            einops.rearrange(logits, 'b c v -> (b c) v'),
                            einops.rearrange(data_y, 'b c -> (b c)')
                        )

                        start_time_backward = timeit.default_timer()
                        torch.cuda.synchronize()
                        backward_pass(loss)
                        torch.cuda.synchronize()
                        end_time_backward = timeit.default_timer()
                        
                        if implementation == "triton":
                            backward_times.append(triton.testing.do_bench(backward_fn, warmup=10000, rep=10000, return_mode='mean'))
                        else:
                            backward_times.append((end_time_backward - start_time_backward) * 1000)

                        end_to_end_times.append((end_time_backward - start_time_forward) * 1000)



                    # save the mean time for forward and backward passes in a pandas dataframe with hyperparameters in the filename
                    df = pd.DataFrame({
                        "implementation": implementation,
                        "d_model": d_model,
                        "seq_len": context_length,
                        "use_mixed_precision": use_mixed_precision,
                        "forward_times_mean": [f"{sum(forward_times) / len(forward_times):.4f} ms"],
                        "backward_times_mean": [f"{sum(backward_times) / len(backward_times):.4f} ms"],
                        "end_to_end_times_mean": [f"{sum(end_to_end_times) / len(end_to_end_times):.4f} ms"],
                        "forward_times_std": [f"{np.std(forward_times):.4f} ms"],
                        "backward_times_std": [f"{np.std(backward_times):.4f} ms"],
                        "end_to_end_times_std": [f"{np.std(end_to_end_times):.4f} ms"],

                    })
                
                # save forward and backward times to a file with hyperparameters in the filename
                os.makedirs("flash_benchmarking", exist_ok=True)

                # save the dataframe to a csv file
                df.to_csv(f"flash_benchmarking/{implementation}_dmodel_{d_model}_context_length_{context_length}_use_mixed_precision_{use_mixed_precision}.csv", index=False)

                # save the dataframe to a latex table file
                # df.to_latex(f"benchmark/{name}_num_warmups_{num_warmups}.txt", index=False)


                
                # except Exception as e:
                #     # empty dataframe
                #     df = pd.DataFrame()
                #     df.to_csv(f"flash_benchmarking/{implementation}_dmodel_{d_model}_context_length_{context_length}_use_mixed_precision_{use_mixed_precision}.csv", index=False)

                #     print(f"Error: {e}")


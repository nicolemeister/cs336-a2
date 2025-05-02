import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import numpy as np

'''
Backend + device type: Gloo + CPU, NCCL + GPU.
all-reduce data size: float32 data tensors ranging over 1MB, 10MB, 100MB, 1GB.
Number of processes: 2, 4, or 6 processes.

'''

def setup(rank, world_size, device_type):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if device_type == "gpu":
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size, data_size, device_type, answer):
    setup(rank, world_size, device_type)
    # Calculate number of elements needed for the desired size in bytes
    num_elements = data_size // 4  # 4 bytes per float32 element
    
    # Create data on the appropriate device
    if device_type == "cuda":
        device = torch.device(f"cuda:{rank}")
        data = torch.randint(0, 10, (num_elements,), dtype=torch.float32, device=device)
    else:
        data = torch.randint(0, 10, (num_elements,), dtype=torch.float32)
    
    # Synchronize before timing
    if device_type == "cuda":
        torch.cuda.synchronize()
    
    # Measure the time for all_reduce operation
    start_time = timeit.default_timer()
    dist.all_reduce(data, async_op=False)
    
    # Synchronize after operation to ensure accurate timing
    if device_type == "cuda":
        torch.cuda.synchronize()
    
    end_time = timeit.default_timer()
    
    # Calculate local timing
    local_time = end_time - start_time
    
    # Gather timing results from all ranks
    all_times = [torch.zeros(1, dtype=torch.float32, device=data.device if device_type == "cuda" else None) for _ in range(world_size)]
    local_time_tensor = torch.tensor([local_time], dtype=torch.float32, device=data.device if device_type == "cuda" else None)
    dist.all_gather(all_times, local_time_tensor)
    
    # Calculate average time across all ranks
    # Each rank sends its local_time_tensor to all other ranks
    # Each rank receives all the timing results from other ranks
    if rank == 0:
        # Move tensors to CPU for calculation if needed
        if device_type == "cuda":
            all_times = [t.cpu() for t in all_times]
        avg_time = sum([t.item() for t in all_times]) / world_size
        # Update the shared tensor with the average time
        answer.copy_(torch.tensor([avg_time], dtype=torch.float32))
    return None

    # print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":

    import pandas as pd
    import argparse
    
    # # Create a list to store results
    # results = []
    # for device_type in ["cuda"]:
    # #     for data_size_name in ['1MB']:
    # #          for world_size in [2]:
    # #  for device_type in ["cpu", "cuda"]:
    #     for data_size_name in ['1MB', '10MB', '100MB', '1GB']:
    #         for world_size in [2, 4, 6]:


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size_name", type=str, default="1MB")
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()

    data_size_name = args.data_size_name
    world_size = args.world_size
    
    device_type = "cuda"

    if device_type == "cuda":
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"Rank {local_rank} using GPU {local_rank}")


    if data_size_name == '1MB': 
        data_size = 1024*1024
    elif data_size_name == '10MB':
        data_size = 1024*1024*10
    elif data_size_name == '100MB':
        data_size = 1024*1024*100
    elif data_size_name == '1GB':
        data_size = 1024*1024*102
    
    # warmup 
    for _ in range(5):
        if device_type == 'cuda':
            torch.cuda.synchronize()

        answer = torch.tensor([0], dtype=torch.float32).share_memory_()
        # Use return_results=True to get the return values from processes
        mp.spawn(fn=distributed_demo, args=(world_size, data_size, device_type, answer), nprocs=world_size, join=True)
        if device_type == 'cuda':
            torch.cuda.synchronize()

    results = []
    timings = []
    for _ in range(10):
        if device_type == 'cuda':
            torch.cuda.synchronize()
        

        answer = torch.tensor([0], dtype=torch.float32).share_memory_()
        # Run the distributed operation
        mp.spawn(fn=distributed_demo, args=(world_size, data_size, device_type, answer), nprocs=world_size, join=True)
        
        if device_type == 'cuda':
            torch.cuda.synchronize()
        
        # Get the result from the shared tensor
        timings.append(answer.item())
    
    print(timings)
    # Store result
    results = []
    results.append({
        'device_type': device_type,
        'data_size': data_size_name,
        'world_size': world_size,
        'time_seconds': np.mean(timings)
    })

    # Convert results to DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv(f'results/distributed_communication_single_node_data_size_{data_size_name}_world_size_{world_size}.csv', index=False)

    # # change the column names to remove the dashes
    # df.columns = ['Backend', 'Data Size', 'Processes', 'Time (s)']

    # # df to latex table
    # df.to_latex(f"results/table_single_node_communication_gpu.txt", index=False)
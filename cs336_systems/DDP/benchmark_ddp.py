from cs336_basics.configs.config import Config
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.training.optimizer import AdamW
from cs336_basics.model.util import crossEntropyLoss, load_model, save_model, get_batch, log_validation_loss, log, build_model
from cs336_basics.data import get_batch
from cs336_systems.DDP.ddp import NaiveDDP, DDPIndividualParameters, BucketedDDP
# from cs336_systems.sharded_optim import Sharded_Optimizer
from cs336_systems.config import Systems_Config
from cs336_systems.ddp import DDP_Bucketed, DDP_Individual_Parameters, DDP_Naive
import torch
import numpy as np
import os
from datetime import timedelta
import torch.distributed as dist
import timeit
import argparse

SEED = 32
BACKEND = "nccl"
DEVICE = "cuda"
NUM_EXP = 10

def setup(world_size):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    # world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]

    dist.init_process_group(BACKEND, rank=rank, world_size=world_size)
    return rank, world_size, local_rank, local_world_size


def train_model(version: str, type: str, world_size, max_mb, sharded):


    # define the xl model 
    d_model = 1600
    d_ff = 6400
    num_layers = 48
    num_heads = 25
    batch_size = 12
    vocab_size = 10000
    rope_theta = 10000.0

    batch_size = 4
    context_length = 128


    # local copy of a model 

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    model.train()

    # if sharded:
    #     optimizer = Sharded_Optimizer(
    #     params=model.parameters(),
    #     optimizer_cls=AdamW,
    #     lr=0.01
    #     )
    # else:
    optimizer = AdamW(
        params=model.parameters(),
        lr=0.001
    )

    if type == "naive":
        model = NaiveDDP(model)
    elif type == "indiv":
        model = DDPIndividualParameters(model)
    elif type == "bucket":
        model = DDP_Bucketed(model, bucket_size_mb=max_mb)

    MINI_BATCH_SIZE = batch_size // world_size

    assert batch_size % world_size == 0, "batch_size must divide world_size"

    forward_time = []
    all_reduce_time = []
    optimizer_step_time = []


    # Generate the full batch first
    full_dataset = torch.randint(0, vocab_size, (vocab_size,)).numpy()
    full_x, full_y = cs336_data.get_batch(
        dataset=full_dataset,
        batch_size=batch_size,  # Get full batch
        context_length=context_length,
        device=device
    )

    # Split the batch into world_size parts
    # Each GPU gets its own slice of the batch
    rank = dist.get_rank()
    x = full_x[rank * MINI_BATCH_SIZE : (rank + 1) * MINI_BATCH_SIZE]
    y = full_y[rank * MINI_BATCH_SIZE : (rank + 1) * MINI_BATCH_SIZE]
    # do some warmup 
    for i in range(5):
        optimizer.zero_grad()
        
        y_hat = model(x)
        loss = crossEntropyLoss(y, y_hat).mean()
        loss.backward()

        torch.cuda.synchronize()
        model.finish_gradient_synchronization()

        torch.cuda.synchronize()

        optimizer.step()

        torch.cuda.synchronize()

    for i in range(NUM_EXP):
        optimizer.zero_grad()
        start_time = timeit.default_timer()
        y_hat = model(x)
        loss = crossEntropyLoss(y, y_hat).mean()
        loss.backward()

        torch.cuda.synchronize()
        forward_time.append(timeit.default_timer()-start_time)

        start_time = timeit.default_timer()

        model.finish_gradient_synchronization()

        torch.cuda.synchronize()
        all_reduce_time.append(timeit.default_timer()-start_time)

        start_time = timeit.default_timer()
        optimizer.step()

        torch.cuda.synchronize()
        optimizer_step_time.append(timeit.default_timer()-start_time)

    if dist.get_rank() == 0:
        f = sum(forward_time)/len(forward_time)
        ar = sum(all_reduce_time)/len(all_reduce_time)
        b = sum(optimizer_step_time)/len(optimizer_step_time)
        t = f + ar + b
        print("FORward:", f)
        print("all reduce: ", ar)
        print("optimizer: ", b)

        # save this into a df and then csv
        df = pd.DataFrame({
            "forward": forward_time,
            "all_reduce": all_reduce_time,
            "optimizer_step": optimizer_step_time,
            "total": t,
            "type": type,
            "world_size": world_size,
            "max_mb": max_mb,
            "sharded": sharded
        })
        df.to_csv(f"ddp_benchmark/type_{type}_world_size_{world_size}_max_mb_{max_mb}_sharded_{sharded}.csv", index=False)




def main():
    parser = argparse.ArgumentParser(description='Train a model with float16 parameters.')
    parser.add_argument('--type', type=str, default=None, help='Version Number of Model')
    parser.add_argument('--max', type=int, default=None, help='max for bucketed version')
    parser.add_argument('--sharded', action="store_true", help='sharded version')
    args = parser.parse_args()

    rank, world_size, local_rank, local_world_size = setup()
    # When running multi-GPU jobs, make sure that different ranks use different GPUs. One method for
    # doing this is to call torch.cuda.set_device(rank)) in the setup function, so that tensor.to("cuda")
    # will automatically send the tensor to the correct GPU.
    torch.cuda.set_device(local_rank % local_world_size)
    train_model(args.type, world_size, args.max, args.sharded)


if __name__ == "__main__":
    main()
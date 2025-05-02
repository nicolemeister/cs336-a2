#!/bin/bash
#SBATCH --job-name=test_hello_batch
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH -c 1
#SBATCH --mem=200G
#SBATCH --gres=gpu:6
#SBATCH --time=00:05:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err



# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_size_name)
      data_size_name="$2"
      shift 2
      ;;
    --world_size)
      world_size="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

uv run python distributed_hello_world.py --data_size_name $data_size_name --world_size $world_size

# uv run nsys profile -o --pytorch nsys_rep/result_${model_size}_context_len_${CONTEXT_LENGTH} python nsys_profile.py $model_size $D_MODEL $D_FF $NUM_LAYERS $NUM_HEADS $NUM_WARMUPS $CONTEXT_LENGTH $NUM_TRIALS 

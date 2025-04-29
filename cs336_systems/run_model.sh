#!/bin/bash
#SBATCH --job-name=test_hello_batch
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH -c 1
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --name)
      model_size="$2"
      shift 2
      ;;
    --d_model)
      D_MODEL="$2"
      shift 2
      ;;
    --d_ff)
      D_FF="$2"
      shift 2
      ;;
    --num_layers)
      NUM_LAYERS="$2"
      shift 2
      ;;
    --num_heads)
      NUM_HEADS="$2"
      shift 2
      ;;
    --num_warmups)
      NUM_WARMUPS="$2"
      shift 2
      ;;
    --context_length)
      CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --num_trials)
      NUM_TRIALS="$2"
      shift 2
      ;;
    --use_mixed_precision)
      use_mixed_precision="$2"
      shift 2
      ;;
    --memory_profile)
      memory_profile="$2"
      shift 2
      ;;
    --compile_model)
      compile_model="$2"
      shift 2
      ;;
    --compile_attention)
      compile_attention="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

echo "Running model with name: $model_size, d_model: $D_MODEL, d_ff: $D_FF, num_layers: $NUM_LAYERS, num_heads: $NUM_HEADS, num_warmups: $NUM_WARMUPS, context_length: $CONTEXT_LENGTH, num_trials: $NUM_TRIALS, use_mixed_precision: $use_mixed_precision, memory_profile: $memory_profile"
# uv run python benchmarking.py --name "$model_size" --d_model "$D_MODEL" --d_ff "$D_FF" --num_layers "$NUM_LAYERS" --num_heads "$NUM_HEADS" --num_warmups "$NUM_WARMUPS" --context_length "$CONTEXT_LENGTH" --num_trials "$NUM_TRIALS" --use_mixed_precision "$use_mixed_precision" --memory_profile "$memory_profile"
uv run python pytorch_attention.py --name "$model_size" --d_model "$D_MODEL" --d_ff "$D_FF" --num_layers "$NUM_LAYERS" --num_heads "$NUM_HEADS" --num_warmups "$NUM_WARMUPS" --context_length "$CONTEXT_LENGTH" --num_trials "$NUM_TRIALS" --use_mixed_precision "$use_mixed_precision" --memory_profile "$memory_profile" --compile_model "$compile_model" --compile_attention "$compile_attention"

uv run python pytorch_attention.py --name "$model_size" --d_model "$D_MODEL" --d_ff "$D_FF" --num_layers "$NUM_LAYERS" --num_heads "$NUM_HEADS" --num_warmups "$NUM_WARMUPS" --context_length "$CONTEXT_LENGTH" --num_trials "$NUM_TRIALS" --use_mixed_precision "$use_mixed_precision" --memory_profile "$memory_profile" --compile_model "0" --compile_attention "1"

uv run python pytorch_attention.py --name "$model_size" --d_model "$D_MODEL" --d_ff "$D_FF" --num_layers "$NUM_LAYERS" --num_heads "$NUM_HEADS" --num_warmups "$NUM_WARMUPS" --context_length "$CONTEXT_LENGTH" --num_trials "$NUM_TRIALS" --use_mixed_precision "$use_mixed_precision" --memory_profile "$memory_profile" --compile_model "1" --compile_attention "0"

# uv run nsys profile -o --pytorch nsys_rep/result_${model_size}_context_len_${CONTEXT_LENGTH} python nsys_profile.py $model_size $D_MODEL $D_FF $NUM_LAYERS $NUM_HEADS $NUM_WARMUPS $CONTEXT_LENGTH $NUM_TRIALS 

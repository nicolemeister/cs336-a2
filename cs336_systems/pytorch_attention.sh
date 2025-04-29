#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Define model configurations
declare -A models
models["small"]="3072 12 1"
# Run each model configuration
# for model_size in "small" "medium" "large" "xl" "2.7B"; do
# for context_length in 128 256 512; do
for model_size in "small"; do
    # for D_MODEL in 16 32 64 128; do
    for D_MODEL in 16; do
        for SEQ_LEN in 256 1024 4096 8192 16384; do
            # Get the parameters for this model size
            params=(${models[$model_size]})
            D_FF=${params[0]}
            NUM_LAYERS=${params[1]}
            NUM_HEADS=${params[2]}
            
            echo "Submitting job for model size: $model_size"

            # Number of warmup steps
            NUM_WARMUPS=5

            echo "Parameters: name = $model_size, d_model=$D_MODEL, d_ff=$D_FF, num_layers=$NUM_LAYERS, num_heads=$NUM_HEADS, num_warmups=$NUM_WARMUPS, context_length=$context_length" 

            # Submit the job
            sbatch run_model.sh --name $model_size --d_model $D_MODEL --d_ff $D_FF --num_layers $NUM_LAYERS --num_heads $NUM_HEADS --num_warmups $NUM_WARMUPS --context_length $SEQ_LEN --num_trials 100 --use_mixed_precision 0 --memory_profile 0 --compile_model 0 --compile_attention 0

            # # Submit the job with compiled attention
            # sbatch run_model.sh --name $model_size --d_model $D_MODEL --d_ff $D_FF --num_layers $NUM_LAYERS --num_heads $NUM_HEADS --num_warmups $NUM_WARMUPS --context_length $SEQ_LEN --num_trials 100 --use_mixed_precision 0 --memory_profile 0 --compile_model 0 --compile_attention 1

            # # Submit the job with compiled model
            # sbatch run_model.sh --name $model_size --d_model $D_MODEL --d_ff $D_FF --num_layers $NUM_LAYERS --num_heads $NUM_HEADS --num_warmups $NUM_WARMUPS --context_length $SEQ_LEN --num_trials 100 --use_mixed_precision 0 --memory_profile 0 --compile_model 1 --compile_attention 0

        done
    done
done
echo "All jobs submitted!"
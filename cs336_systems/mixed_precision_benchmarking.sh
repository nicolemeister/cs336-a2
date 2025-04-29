#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Define model configurations
declare -A models
models["small"]="768 3072 12 12"
models["medium"]="1024 4096 24 16"
models["large"]="1280 5120 36 20"
models["xl"]="1600 6400 48 25"
models["2.7B"]="2560 10240 32 32"


# Run each model configuration
for model_size in "small" "medium" "large" "xl" "2.7B"; do
    # Get the parameters for this model size
    params=(${models[$model_size]})
    D_MODEL=${params[0]}
    D_FF=${params[1]}
    NUM_LAYERS=${params[2]}
    NUM_HEADS=${params[3]}
    
    echo "Submitting job for model size: $model_size"

    # Number of warmup steps
    NUM_WARMUPS=5

    echo "Parameters: name = $model_size, d_model=$D_MODEL, d_ff=$D_FF, num_layers=$NUM_LAYERS, num_heads=$NUM_HEADS, num_warmups=$NUM_WARMUPS" 

    # Submit the job
    sbatch run_model.sh $model_size $D_MODEL $D_FF $NUM_LAYERS $NUM_HEADS $NUM_WARMUPS 128 10 1
    
done

echo "All jobs submitted!"

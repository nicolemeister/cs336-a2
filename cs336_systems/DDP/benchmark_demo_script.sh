#!/bin/bash

    #     for data_size_name in ['1MB', '10MB', '100MB', '1GB']:
    #         for world_size in [2, 4, 6]:

# Run each model configuration
# for model_size in "small" "medium" "large" "xl" "2.7B"; do
# for context_length in 128 256 512; do
for data_size_name in '1MB' '10MB' '100MB' '1GB'; do
    for world_size in 2 4 6; do
        # Get the parameters for this model size
        # Submit the job
        sbatch gpu.sh --data_size_name $data_size_name --world_size $world_size
        
    done
done
echo "All jobs submitted!"

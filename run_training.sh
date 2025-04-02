#!/bin/bash

# Set default dataset path
DEFAULT_DATASET="examples/kitchen"

# Check if dataset argument is provided
if [ -z "$1" ]; then
    echo "No dataset provided, using default: $DEFAULT_DATASET"
    DATASET=$DEFAULT_DATASET
else
    DATASET=data/$1
    gcloud storage rsync -r gs://tour_storage/$DATASET $DATASET
fi

# Run the training
echo "Running vggt"

python vggt_to_colmap.py --image_dir $DATASET/images --output_dir $DATASET/sparse_vggt --binary


# Sync results back to Google Cloud Storage only if dataset argument was provided
if [ ! -z "$1" ]; then
    gcloud storage rsync -r $DATASET gs://tour_storage/$DATASET 
fi 
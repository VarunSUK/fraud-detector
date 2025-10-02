#!/bin/bash
set -e

# Entrypoint script for Python ML training pipeline

echo "Starting fraud detection ML training pipeline..."

# Default to training if no command specified
if [ $# -eq 0 ]; then
    echo "No command specified, running training..."
    python train.py --generate-data --model-type ensemble
else
    echo "Running command: $@"
    exec "$@"
fi




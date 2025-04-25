#!/bin/bash
set -e

# Verify model exists
if [ ! -f "/opt/render/project/src/models/dog_disease_model_96.h5" ]; then
    echo "Model file not found, attempting to download..."
    gdown "https://drive.google.com/uc?id=1dh2J-arVsnBJA7xVRZP9r1e4x8qPUeAc" -O /opt/render/project/src/models/dog_disease_model_96.h5
fi

# Set Python path
export PYTHONPATH=$PYTHONPATH:/opt/render/project/src

# Start the application
cd VetScope && gunicorn VetScope.app:app --bind 0.0.0.0:$PORT 
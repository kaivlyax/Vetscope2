#!/bin/bash
set -e

# Install requirements
pip install -r requirements.txt

# Create models directory
mkdir -p /opt/render/project/src/models

# Download model file
gdown "https://drive.google.com/uc?id=1dh2J-arVsnBJA7xVRZP9r1e4x8qPUeAc" -O /opt/render/project/src/models/dog_disease_model_96.h5

# Verify download
if [ -f "/opt/render/project/src/models/dog_disease_model_96.h5" ]; then
    echo "Model file downloaded successfully"
    ls -lh /opt/render/project/src/models/
else
    echo "Failed to download model file"
    exit 1
fi 
#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Install additional packages with mim
mim install mmengine
mim install "mmcv>=2.0.1"

# Clone the PoseEstimation repository
git clone https://github.com/iamNavid1/PoseEstimation.git

# Navigate to the PoseEstimation directory
cd PoseEstimation

# Install dependencies for the PoseEstimation project
pip install -r requirements.txt

# Install the PoseEstimation project in editable mode
pip install -v -e .

# Navigate back to the VisionPipeline directory
cd ..


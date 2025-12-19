#!/bin/bash
# Build and run JPEG-AI Docker container for EuroSAT compression

set -e

# Add Docker to PATH if not already there (macOS Docker Desktop)
if ! command -v docker &> /dev/null; then
    if [ -f "/Applications/Docker.app/Contents/Resources/bin/docker" ]; then
        export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
    fi
fi

# Configuration
IMAGE_NAME="jpeg-ai-eurosat"
CONTAINER_NAME="jpeg-ai-compression"
DATA_DIR="$(pwd)/data/EuroSAT_RGB"
OUTPUT_DIR="$(pwd)/data/eurosat_jpeg_ai_compressed"
SCRIPTS_DIR="$(pwd)/docker/jpeg-ai"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}JPEG-AI Docker Setup for EuroSAT${NC}"
echo -e "${BLUE}======================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

# Build Docker image
echo -e "\n${GREEN}Step 1: Building Docker image...${NC}"
cd docker/jpeg-ai
docker build -t $IMAGE_NAME .
cd ../..

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run compression in container
echo -e "\n${GREEN}Step 2: Running compression in container...${NC}"
echo -e "Data: $DATA_DIR"
echo -e "Output: $OUTPUT_DIR"

# Check if NVIDIA GPU is available (Linux with nvidia-docker)
if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    GPU_FLAG="--gpus all"
    echo -e "Using NVIDIA GPU acceleration"
else
    GPU_FLAG=""
    echo -e "Running on CPU (no NVIDIA GPU detected)"
fi

docker run --rm \
    --name $CONTAINER_NAME \
    $GPU_FLAG \
    -v "$DATA_DIR:/data/input:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -v "$SCRIPTS_DIR:/scripts" \
    $IMAGE_NAME \
    python3 /scripts/compress_eurosat.py

echo -e "\n${GREEN}Done! Compressed images saved to:${NC}"
echo -e "$OUTPUT_DIR"

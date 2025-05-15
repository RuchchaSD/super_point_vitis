#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

source /etc/profile.d/pynq_venv.sh

source debug.sh

# SuperPoint Segmentation Benchmark Script
# Allows easy configuration and execution of segmentation benchmarking

# Default values (can be changed below or passed as environment variables)
IMAGE_PATH=${IMAGE_PATH:-"/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis/temp/img/GettyImages-496309064_resized.jpg"}  # Default test image path
SERVER_URL=${SERVER_URL:-"http://192.248.10.70:8000/segment"}  # Default segmentation server URL
ITERATIONS=${ITERATIONS:-100}                       # Number of iterations for benchmarking
DEBUG=${DEBUG:-0}                                  # Enable debug output (0=off, 1=on)

# Print configured values
echo "SuperPoint Segmentation Benchmark"
echo "===================================="
echo "Image Path: ${IMAGE_PATH}"
echo "Server URL: ${SERVER_URL}"
echo "Iterations: ${ITERATIONS}"
echo "Debug Mode: ${DEBUG}"
echo ""

# Check if the image file exists
if [ ! -f "${IMAGE_PATH}" ]; then
    echo "ERROR: Image file not found: ${IMAGE_PATH}"
    echo "Please specify a valid image path using the IMAGE_PATH environment variable:"
    echo "Example: IMAGE_PATH=/path/to/image.png ./run_segmentation.sh"
    exit 1
fi

# Suggest alternative test images
echo "Available test images you can try (if segmentation fails with current image):"
find temp/img -type f -name "*.png" -o -name "*.jpg" | head -5
echo ""

# Build command line arguments
CMD="./bin/segment_image"

# Add image path, server URL, and iterations
CMD="${CMD} ${IMAGE_PATH} ${SERVER_URL} ${ITERATIONS}"

# Enable debug mode if requested
if [ "${DEBUG}" -eq 1 ]; then
    echo "Debug mode enabled. Setting DEBUG_SEGMENT environment variable."
    export DEBUG_SEGMENT=1
fi

# Print the command that will be executed
echo "Executing: ${CMD}"
echo "===================================="
echo "Running benchmark with ${ITERATIONS} iterations..."
echo ""

# Execute the command
eval ${CMD}

echo "Done!"
# End of script 
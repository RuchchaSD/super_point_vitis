#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

source /etc/profile.d/pynq_venv.sh

# SuperPoint Demo Runner Script
# Allows easy configuration of SuperPoint demo execution parameters

# Default values (can be changed below or passed as environment variables)
IMPLEMENTATION=${IMPLEMENTATION:-"multi"}  # "single" or "multi"
RUNNERS=${RUNNERS:-10}                      # Number of DPU runners (only used with multi-threaded)
ITERATIONS=${ITERATIONS:-100}                # Number of inference iterations to run
MODEL=${MODEL:-"/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis/superpoint_tf.xmodel"}     # Model file name
IMAGE=${IMAGE:-"/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis/temp/imgs/1403636669163555584.png"}                 # Input image file name
OUTPUT_DIR=${OUTPUT_DIR:-"./results"}      # Directory to store results

# Print configured values
echo "SuperPoint Demo Runner"
echo "======================"
echo "Implementation: ${IMPLEMENTATION} (multi=multi-threaded, single=single-threaded)"
if [ "$IMPLEMENTATION" == "multi" ]; then
    echo "Number of DPU runners: ${RUNNERS}"
fi
echo "Iterations: ${ITERATIONS}"
echo "Model: ${MODEL}"
echo "Image: ${IMAGE}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Build command line arguments
CMD="./build/demo"

# Add implementation flag
if [ "$IMPLEMENTATION" == "single" ]; then
    CMD="${CMD} -s"
fi

# Add runners flag if multi-threaded
if [ "$IMPLEMENTATION" == "multi" ]; then
    CMD="${CMD} -t ${RUNNERS}"
fi

# Add iterations flag if more than 1
if [ $ITERATIONS -gt 1 ]; then
    CMD="${CMD} -i ${ITERATIONS}"
fi

# Add model and image
CMD="${CMD} ${MODEL} ${IMAGE}"

# Print the command that will be executed
echo "Executing: ${CMD}"
echo "======================"

# Execute the command
eval ${CMD}

# Copy result files to output directory
echo "Copying results to ${OUTPUT_DIR}"
for img in result_superpoint_*.jpg; do
    if [ -f "$img" ]; then
        cp "$img" "${OUTPUT_DIR}/"
        rm "$img"  # Remove the original result file
        echo "Copied $img to ${OUTPUT_DIR}/"
    fi
done

echo "Done!"
# End of script
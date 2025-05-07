#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

source /etc/profile.d/pynq_venv.sh

# SuperPoint Benchmark Script
# Default values (can be changed below or passed as environment variables)
THREADS=${THREADS:-4}                  # Number of pre/post-processing threads

#MODEL="/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis/superpoint_tf.xmodel"

MODEL="/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis/compiled_SP_by_H.xmodel"  # Model file name
INPUT_DIR=${INPUT_DIR:-"/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis/i_ajuntament"}      # Directory with input images
OUTPUT_DIR=${OUTPUT_DIR:-"./results/benchmark"}  # Directory to store results
FILE_EXT=${FILE_EXT:-"ppm"}            # File extension to filter input images
HOMOGRAPHY=${HOMOGRAPHY:-true}         # Whether to run homography test

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Print configured values
echo "SuperPoint Benchmark"
echo "===================="
echo "Model: ${MODEL}"


# Build command line arguments
CMD="./build/superpoint_benchmark -h"

# Add threads parameter
CMD="${CMD} -t ${THREADS}"

# Add file extension filter
CMD="${CMD} -f ${FILE_EXT}"

# Add output directory
CMD="${CMD} -o ${OUTPUT_DIR}"

# Add homography flag if enabled
if [ "${HOMOGRAPHY}" = "true" ]; then
    CMD="${CMD} -h"
fi

# Add model and input directory
CMD="${CMD} ${MODEL} ${INPUT_DIR}"

# Print the command that will be executed
echo "Executing: ${CMD}"
echo "===================="

# Execute the command
eval ${CMD}

echo "Done!"
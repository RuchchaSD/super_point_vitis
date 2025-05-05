#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

source /etc/profile.d/pynq_venv.sh

# SuperPoint Feature Extraction Script
# Extracts and saves keypoints and descriptors to separate folders

# Default values (can be changed below or passed as environment variables)
THREADS=${THREADS:-4}                        # Number of pre/post-processing threads
MODEL=${MODEL:-"/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis/compiled_SP_by_H.xmodel"}  # Model file name
INPUT_DIR=${INPUT_DIR:-"/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis/temp/imgs"}  # Directory with input images
OUTPUT_DIR=${OUTPUT_DIR:-"./feature_outputs/SP_H"}       # Base directory for feature storage
FILE_EXT=${FILE_EXT:-"png"}                  # File extension to filter input images

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Print configured values
echo "SuperPoint Feature Extraction"
echo "============================="
echo "Pre/post-processing threads: ${THREADS}"
echo "Model: ${MODEL}"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "File extension: ${FILE_EXT}"
echo ""

# Count input files
NUM_FILES=$(find ${INPUT_DIR} -name "*.${FILE_EXT}" | wc -l)
echo "Found ${NUM_FILES} ${FILE_EXT} files in ${INPUT_DIR}"
echo ""

# Build command line arguments
CMD="./build/extract_features"

# Add threads parameter
CMD="${CMD} -t ${THREADS}"

# Add file extension filter
CMD="${CMD} -f ${FILE_EXT}"

# Add model, input and output directories
CMD="${CMD} ${MODEL} ${INPUT_DIR} ${OUTPUT_DIR}"

# Print the command that will be executed
echo "Executing: ${CMD}"
echo "============================="

# Execute the command
eval ${CMD}

echo "Done!"
# End of script
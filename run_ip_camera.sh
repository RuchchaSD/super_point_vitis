#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

source /etc/profile.d/pynq_venv.sh

source debug.sh

export USE_SEGMENTATION_MASK=1
SEGMENTER_URL="http://192.248.10.70:8000/segment"

# SuperPoint IP Camera Processing Demo Runner Script
# Allows easy configuration and execution of SuperPoint IP camera demo

# Default values (can be changed below or passed as environment variables)
THREADS=${THREADS:-2}                        # Number of pre/post-processing threads
MODEL=${MODEL:-"/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis/compiled_SP_by_H.xmodel"}  # Model file name
IP_ADDRESS=${IP_ADDRESS:-"192.168.8.111"}    # IP address of the camera stream
PORT=${PORT:-"8080"}                         # Port of the camera stream 
FPS=${FPS:-"15"}                             # Target FPS for frame capture
PROTOCOL=${PROTOCOL:-"rtsp"}                 # Protocol for video streaming (rtsp, http, etc.)
CAMERA_FPS=${CAMERA_FPS:-"30"}               # Actual camera FPS (default: 30)

# Print configured values
echo "SuperPoint IP Camera Processing Demo"
echo "===================================="
echo "Pre/post-processing threads: ${THREADS}"
echo "Model: ${MODEL}"
echo "IP Address: ${IP_ADDRESS}"
echo "Port: ${PORT}"
echo "Target FPS: ${FPS}"
echo "Protocol: ${PROTOCOL}"
echo "Camera FPS: ${CAMERA_FPS}"
echo ""

# Build command line arguments
CMD="./bin/demo_ip_camera"

# Add threads parameter
CMD="${CMD} -t ${THREADS}"

# Add model, IP address, port, fps and protocol
CMD="${CMD} ${MODEL} ${IP_ADDRESS} ${PORT} ${FPS} ${PROTOCOL} ${CAMERA_FPS}"

# Print the command that will be executed
echo "Executing: ${CMD}"
echo "===================================="
echo "Press 'q' in the display window to exit"
echo ""

# Execute the command
eval ${CMD}

echo "Done!"
# End of script 
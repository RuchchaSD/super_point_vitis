#!/bin/bash

# Create directory for received masks
mkdir -p received_masks

# Run curl command and save raw response
curl -X POST \
  -F "image=@/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis/People in park.jpg" \
  http://192.248.10.70:8000/segment \
  -o response.json \
  -w "\nTotal time: %{time_total} seconds\n"

# Process the response and decode base64 mask
python3 - << 'EOF'
import json
import base64
import os

# Read the JSON response
with open('response.json', 'r') as f:
    response_data = json.load(f)

# Get the base64 encoded mask
mask_b64 = response_data.get('merged_mask')

if mask_b64:
    # Save the decoded mask
    output_path = "received_masks/merged_mask.png"
    with open(output_path, "wb") as out_file:
        out_file.write(base64.b64decode(mask_b64))
    print("✅ Received and saved merged mask as:", output_path)
else:
    print("⚠️ No person mask found in the image.")

# Clean up temporary response file
os.remove('response.json')
EOF

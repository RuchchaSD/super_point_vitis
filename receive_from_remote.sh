#!/bin/bash

set -e

git pull

REMOTE_URL="root@192.248.10.70"
REMOTE_PATH="/mnt/kr260/root/jupyter_notebooks/Fyp/sp_cmake/super_point_vitis"

REMOTE_TARGET="$REMOTE_URL:$REMOTE_PATH"

# Rsync and SSH options
RSYNC_OPTS="-avzP --partial --inplace"
SSH_OPTS="ssh -T -c aes128-gcm@openssh.com \
    -o Compression=no \
    -o ControlMaster=auto \
    -o ControlPath=~/.ssh/cm-%r@%h:%p \
    -o ControlPersist=600"

# Sync all directories and files
echo "Syncing entire project..."
rsync $RSYNC_OPTS -e "$SSH_OPTS" --delete "$REMOTE_TARGET/" ./

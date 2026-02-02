#!/bin/bash

# This script fixes the dark camera image issue on macOS by disabling auto-exposure priority.
# It uses the uvc-util tool.

set -e

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "This script is only for macOS."
    exit 1
fi

# Create a temporary directory for uvc-util if it doesn't exist
UVC_UTIL_DIR=".cache/uvc-util"
mkdir -p .cache

if [ ! -d "$UVC_UTIL_DIR" ]; then
    echo "Cloning uvc-util..."
    git clone https://github.com/jtfrey/uvc-util.git "$UVC_UTIL_DIR"
    cd "$UVC_UTIL_DIR"
    echo "Building uvc-util..."
    make
    cd -
else
    echo "uvc-util already exists in $UVC_UTIL_DIR"
fi

echo "Applying camera fix..."
"$UVC_UTIL_DIR/uvc-util" -I 0x01140000 -s auto-exposure-priority=1

echo "Camera fix applied successfully!"
echo "If the image is still dark, try listing your devices with: $UVC_UTIL_DIR/uvc-util -l"

#!/bin/bash

# Bus Safety System - Run Script

echo " Starting Bus Safety System..."
echo ""

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    exit 1
fi

# Run main system
echo "Starting main system..."
python3 main.py "$@"


#!/bin/bash

# Check if the package directory exists and contains a setup.py file
if [ -f "/sldp/setup.cfg" ]; then
    echo "Installing Sliding Puzzles..."
    pip3 install -e /sldp
else
    echo "Sliding Puzzles not found. Mount it as a volume at /sldp"
fi

# Execute the original command
exec "$@"
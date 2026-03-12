#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "Installing gem in editable mode..."
uv pip install -e .

echo "Environment setup complete."

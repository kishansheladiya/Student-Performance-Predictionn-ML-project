#!/usr/bin/env bash
set -euo pipefail

echo "Starting training inside container..."
python ml/download_data.py
python ml/train.py
echo "Training completed. Models saved to /app/models"

#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# Put ALL pip installs here, NOT in container.def.
# Only use container.def for the base Docker image and apt-get packages.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# The overlay may be recreated on container rebuilds — a venv inside
# it will be lost and all subsequent jobs will fail.
# Use: uv pip install --system <packages>
# Do NOT create a venv (python -m venv, virtualenv, etc.).

set -e

# Runtime stack for the GaussMark method and reusable evaluation scripts.
# Keep versions moderately pinned so the environment is reproducible without
# relying on whatever happens to be preinstalled on the host.
uv pip install --system \
    "numpy>=1.26,<3" \
    "scipy>=1.12,<2" \
    "scikit-learn>=1.4,<2" \
    "torch>=2.5,<2.7" \
    "transformers>=4.46,<5" \
    "accelerate>=1.1,<2" \
    "safetensors>=0.4.5,<1" \
    "sentencepiece>=0.2,<1"

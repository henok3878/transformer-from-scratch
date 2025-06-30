#!/usr/bin/env bash
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected"
    conda env create -f environment-gpu.yml
else
    echo "No GPU found"
    conda env create -f environment-cpu.yml
fi


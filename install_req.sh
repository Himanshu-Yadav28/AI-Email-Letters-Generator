#!/bin/sh
# install CPU-only torch then other requirements
python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.2.0+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html
python -m pip install --no-cache-dir -r requirements.txt

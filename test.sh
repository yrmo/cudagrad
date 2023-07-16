#!/bin/bash
set -eou pipefail

git submodule update --init --recursive

rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
cmake ..
make
./cudagrad_test

pip uninstall cudagrad
pip cache purge
cd /
pip install cudagrad
cd ~/cudagrad
py tests/test.py
# make clean

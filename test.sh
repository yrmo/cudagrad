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

pip uninstall -y cudagrad
pip cache purge
cd ~/cudagrad
pip install . # hmmm, this tests local install but not pypi install...
py tests/test.py
# make clean

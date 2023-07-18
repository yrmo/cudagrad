#!/bin/bash
set -eou pipefail

if [ $# -eq 0 ]; then
    echo "Error: No argument provided."
    exit 1
fi

if [ "$1" = "lint" ]; then
  py -m mypy --ignore-missing-imports --pretty .
elif [ "$1" = "clean" ]; then
  py -m isort .
  py -m black .
elif [ "$1" = "test" ]; then
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

  cd ~/cudagrad
  c++ -std=c++11 -I./src examples/example.cpp && ./a.out
  pip install cudagrad; py ./examples/example.py
elif [ "$1" = "publish" ]; then
  py -m pip uninstall -y cudagrad
  pip cache purge
  rm -rf dist
  py setup.py sdist
  py -m pip install --upgrade twine
  py -m twine upload dist/*
else
  echo "Error: Invalid command \`$1\`."
  exit 1
fi

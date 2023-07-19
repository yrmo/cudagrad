#!/bin/bash
set -eou pipefail

if [ $# -eq 0 ]; then
    echo "Error: No argument provided."
    exit 1
fi

if [ "$1" = "lint" ]; then
  python -m mypy --ignore-missing-imports --pretty .
elif [ "$1" = "clean" ]; then
  python -m isort .
  python -m black .
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
  python tests/test.py
  # make clean

  cd ~/cudagrad
  c++ -std=c++11 -I./src examples/example.cpp && ./a.out
  pip install cudagrad; python ./examples/example.py
elif [ "$1" = "publish" ]; then
  python -m pip uninstall -y cudagrad
  pip cache purge
  rm -rf dist
  python setup.py sdist
  python -m pip install --upgrade twine
  python -m twine upload dist/*
elif [ "$1" = "bump" ]; then
  if [ "$2" = "major" ]; then
    python -c "import toml; d = toml.load('pyproject.toml'); d['project']['version'] = '.'.join(str(int(x) + 1) if i == 0 else x for i, x in enumerate(d['project']['version'].split('.'))); toml.dump(d, open('pyproject.toml', 'w'))"
  elif [ "$2" = "minor" ]; then
    python -c "import toml; d = toml.load('pyproject.toml'); d['project']['version'] = '.'.join(str(int(x) + 1) if i == 1 else x for i, x in enumerate(d['project']['version'].split('.'))); toml.dump(d, open('pyproject.toml', 'w'))"
  elif [ "$2" = "patch" ]; then
    python -c "import toml; d = toml.load('pyproject.toml'); d['project']['version'] = '.'.join(str(int(x) + 1) if i == 2 else x for i, x in enumerate(d['project']['version'].split('.'))); toml.dump(d, open('pyproject.toml', 'w'))"
  else
    echo "Error: Invalid bump option: \`$2\`."
    exit 1
  fi
else
  echo "Error: Invalid command \`$1\`."
  exit 1
fi

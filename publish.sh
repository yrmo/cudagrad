# untested in full
# https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56
#!/usr/bin/env bash
set -eou pipefail

py -m pip uninstall cudagrad
pip cache purge
rm -rf dist
# py -m pip install --upgrade build
# py -m build
py setup.py sdist
py -m pip install --upgrade twine
py -m twine upload dist/*

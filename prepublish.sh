rm -rf dist
python setup.py sdist bdist_wheel
# TODO fix hardcode version
cd dist
pip install cudagrad-0.1.0.tar.gz
python -c "from cudagrad import hello; hello()"

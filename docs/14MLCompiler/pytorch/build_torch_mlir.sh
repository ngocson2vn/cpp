git clone https://github.com/llvm/torch-mlir.git
cd torch-mlir
git submodule update --init --recursive

# Install pybind11
pip install pybind11 nanobind

# Install the latest ninja version
git clone https://github.com/ninja-build/ninja.git && cd ninja
git checkout release
./configure.py --bootstrap
sudo cp -vf ./ninja /usr/local/bin

CMAKE_GENERATOR=Ninja python setup.py bdist_wheel

pip install dist/torch_mlir-0.0.1-cp311-cp311-linux_x86_64.whl
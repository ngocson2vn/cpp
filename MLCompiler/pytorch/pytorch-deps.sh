# Python
pyenv install 3.11.2
unset http_proxy https_proxy
pip install --upgrade pip
pip install numpy filelock fsspec jinja2 networkx

# cuda-deps
./cuda-deps.sh

# libomp
apt install -y libomp-dev
ln -sf /usr/lib/llvm-14/lib/libomp.so.5 /lib/x86_64-linux-gnu/libomp.so

# pytorch
/usr/local/tao/agent/modules/bvc/bin/bvc clone aml/lagrange/pytorch --version 1.0.0.498

# Verify CUDA version used in a pytorch build job
# Search for 'Found CUDA'
# Result: Found CUDA: /usr/local/cuda (found version "12.8") 

#====================================
# Docker image deps
#====================================
libcudart.so.13
libcupti.so.13
libcudnn.so.9
libnvshmem_host.so.3
libomp.so
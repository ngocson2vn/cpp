# Python
pyenv install 3.11.2
unset http_proxy https_proxy
pip install --upgrade pip
pip install numpy

# cuda-deps
./cuda-deps.sh

# libomp
apt install -y libomp-dev
ln -sf /usr/lib/llvm-14/lib/libomp.so.5 /lib/x86_64-linux-gnu/libomp.so

# pytorch
/usr/local/tao/agent/modules/bvc/bin/bvc clone aml/lagrange/pytorch --version 1.0.0.498

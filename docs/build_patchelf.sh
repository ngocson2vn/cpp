wget https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0.tar.gz
tar -xf patchelf-0.18.0.tar.gz
cd patchelf-0.18.0

./configure --prefix=/usr
make && make install
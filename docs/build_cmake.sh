apt install libssl-dev

# create a `cmake` directory and extract cmake into there
# build cmake in there, and install to prefix

PREFIX=/usr

wget -SL https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0.tar.gz
mkdir -p cmake-3.30.0
tar -xvf cmake-3.30.0.tar.gz --strip-components=1 -C cmake-3.30.0

cd cmake-3.30.0
./bootstrap --prefix=$PREFIX --parallel=`nproc`
nice -n20 make -j`nproc`

sudo nice -n20 make install

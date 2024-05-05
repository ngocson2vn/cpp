# Build
git clone https://github.com/facebook/folly
cd folly
python3 ./build/fbcode_builder/getdeps.py --scratch-path=./tmp --install-prefix=/data00/home/son.nguyen/git/ngocson2vn/cpp/Folly --allow-system-packages build

# Dependencies
sudo apt install libfmt-dev -y
sudo apt-get install libboost-all-dev -y
sudo apt install libgoogle-glog-dev -y
sudo apt install libdouble-conversion-dev -y
sudo apt install libevent-dev -y
sudo apt install libtcl -y

dpkg -L libgoogle-glog-dev
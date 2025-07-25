apt -y install build-essential

wget http://ftp.tsukuba.wide.ad.jp/software/gcc/releases/gcc-8.3.0/gcc-8.3.0.tar.xz
tar -xf gcc-8.3.0.tar.xz
cd gcc-8.3.0
vim ./contrib/download_prerequisites
# Set base_url as below:
# base_url='http://gcc.gnu.org/pub/gcc/infrastructure/'

find -name \*.tar.\* -delete

./contrib/download_prerequisites
./configure --disable-multilib --prefix=/usr
make -j `nproc` && make install


# 8.3.5
wget http://ftp.tsukuba.wide.ad.jp/software/gcc/releases/gcc-8.5.0/gcc-8.5.0.tar.gz

# 9.5.0
wget http://ftp.tsukuba.wide.ad.jp/software/gcc/releases/gcc-9.5.0/gcc-9.5.0.tar.gz

# 10.1.0
wget http://ftp.tsukuba.wide.ad.jp/software/gcc/releases/gcc-10.1.0/gcc-10.1.0.tar.gz
tar -xf gcc-10.1.0.tar.gz
cd gcc-10.1.0
./contrib/download_prerequisites
./configure --disable-multilib --prefix=/usr
make -j `nproc` && make install
# Create symlinks
ls -l /usr/lib64/libstdc++.so.6.0.28
sudo cp -v /usr/lib64/libstdc++.so.6.0.28 /lib/x86_64-linux-gnu/
cd /lib/x86_64-linux-gnu/
sudo ln -sf libstdc++.so.6.0.28 libstdc++.so.6

# 12.5.0
wget https://ftp.tsukuba.wide.ad.jp/software/gcc/releases/gcc-12.5.0/gcc-12.5.0.tar.gz
tar -xzf gcc-12.5.0.tar.gz
cd gcc-12.5.0
./contrib/download_prerequisites
./configure --disable-multilib --prefix=/usr
make -j `nproc` && make install

# Create symlinks
ls -l /usr/lib64/libstdc++.so.6.0.30
sudo cp -v /usr/lib64/libstdc++.so.6.0.30 /lib/x86_64-linux-gnu/
cd /lib/x86_64-linux-gnu/
sudo ln -sf libstdc++.so.6.0.30 libstdc++.so.6


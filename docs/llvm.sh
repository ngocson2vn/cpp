##########################################################################################
# clang
##########################################################################################
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo apt install -y lsb-release wget software-properties-common gnupg

vim ./llvm.sh
export http_proxy=http://[fdbd:dc61:14:39::137]:8585
export https_proxy=http://[fdbd:dc61:14:39::137]:8585
export no_proxy=laniakea.com

sudo ./llvm.sh 17

sudo ln -sf /usr/bin/clang-17 /usr/bin/clang
sudo ln -sf /usr/bin/clang++-17 /usr/bin/clang++
sudo ln -sf /usr/bin/ld.lld-17 /usr/bin/ld.lld

sudo ./llvm.sh 22

sudo ln -sf /usr/bin/clang-22 /usr/bin/clang
sudo ln -sf /usr/bin/clang++-22 /usr/bin/clang++
sudo ln -sf /usr/bin/ld.lld-22 /usr/bin/ld.lld

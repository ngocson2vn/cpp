##########################################################################################
# clang
##########################################################################################
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo apt install -y lsb-release wget software-properties-common gnupg
sudo ./llvm.sh 17

sudo ln -sf /usr/bin/clang-17 /usr/bin/clang
sudo ln -sf /usr/bin/clang++-17 /usr/bin/clang++
sudo ln -sf /usr/bin/ld.lld-17 /usr/bin/ld.lld

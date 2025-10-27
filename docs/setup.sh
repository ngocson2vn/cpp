##########################################################################################
# Python
##########################################################################################
apt install ca-certificates
curl https://pyenv.run | bash
# git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec bash -l

apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# To avoid ensurepip error
export CONFIGURE_OPTS="--with-ensurepip=no"

pyenv install 3.7.3
pyenv global 3.7.3

##########################################################################################
# clang
##########################################################################################
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo apt install -y lsb-release wget software-properties-common gnupg
sudo ./llvm.sh 17

##########################################################################################
# latex
##########################################################################################
sudo apt update
sudo apt upgrade # Sometimes, we must upgrade all packages
sudo apt install -y texlive-full

# compile a latex file
pdflatex layout.tex

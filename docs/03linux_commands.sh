#==========================================================================================
# SSH
#==========================================================================================
ssh -L 9999:127.0.0.1:9999 gpudev_va2

#==========================================================================================
# Install python
#==========================================================================================
sudo apt install ca-certificates
curl https://pyenv.run | bash
# git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec bash -l

sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

pyenv install 3.7.3
pyenv global 3.7.3

pyenv install 3.11 && pyenv global 3.11
python -V
pip install --upgrade pip


#==========================================================================================
# Install kinit
#==========================================================================================
sudo apt install krb5-user
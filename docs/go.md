# Install bison (for Linux only)
sudo apt-get install -y bison golang-go

# Install gvm
bash < <(curl -s -S -L https://raw.githubusercontent.com/moovweb/gvm/master/binscripts/gvm-installer)
source ~/.gvm/scripts/gvm

# Install go
gvm install go1.24.0 --prefer-binary && gvm use go1.24.0 --default
go version
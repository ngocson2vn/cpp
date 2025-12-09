# Install the latest ninja version
git clone https://github.com/ninja-build/ninja.git && cd ninja
git checkout release
./configure.py --bootstrap
sudo cp -vf ./ninja /usr/local/bin

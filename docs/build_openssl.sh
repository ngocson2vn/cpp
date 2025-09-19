# Could not build the ssl module!
# Python requires an OpenSSL 1.0.2 or 1.1 compatible libssl with X509_VERIFY_PARAM_set1_host().
wget https://www.openssl.org/source/openssl-1.0.2o.tar.gz
tar -xzf openssl-1.0.2o.tar.gz
cd openssl-1.0.2o
./config shared --prefix=/usr
make
make install > install.log 2>&1

cp /usr/include/openssl/opensslconf.h /usr/include/x86_64-linux-gnu/openssl/opensslconf.h
cp -vrf /usr/lib64/libcrypto.so.1.0.0 /usr/lib/x86_64-linux-gnu/
cp -vrf /usr/lib64/libssl.so.1.0.0 /usr/lib/x86_64-linux-gnu/

# Verify
python
Python 3.7.3 (default, Jun  6 2024, 00:53:56)
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import ssl
>>> ssl.OPENSSL_VERSION
'OpenSSL 1.0.2o  27 Mar 2018'

# Could not build the ssl module!
# Python requires a OpenSSL 1.1.1 or newer

wget https://www.openssl.org/source/openssl-1.1.1.tar.gz
tar -xzf openssl-1.1.1.tar.gz
cd openssl-1.1.1
./config --prefix=/usr
make && sudo make install

sudo cp -vrf /usr/lib64/libcrypto.so.1.1 /usr/lib/x86_64-linux-gnu/libcrypto.so.1.1
sudo cp -vrf /usr/lib64/libssl.so.1.1 /usr/lib/x86_64-linux-gnu/libssl.so.1.1
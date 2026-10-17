export PATH=/opt/tiger/cpp_tools/x86_64_x86_64_clang_2020/bin:$PATH
wget https://www.python.org/ftp/python/3.7.13/Python-3.7.13.tar.xz \
    && tar -xf Python-3.7.13.tar.xz \
    && cd Python-3.7.13 \
    && ./configure --prefix=/usr --exec_prefix=/usr --enable-optimizations --enable-shared --with-lto --enable-loadable-sqlite-extensions --enable-ipv6 --with-openssl=/usr --disable-test-modules \
    && make -j $(nproc) PROFILE_TASK=none && make altinstall && \
    cd .. && rm -rf Python-3.7.13 Python-3.7.13.tar.xz
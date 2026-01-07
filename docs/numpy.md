# Fix build error
fatal error: numpy/arrayobject.h: No such file or directory
```Bash
export CFLAGS=-I/data00/home/son.nguyen/.pyenv/versions/3.9.0/lib/python3.9/site-packages/numpy/_core/include
export CFLAGS=-I/data05/home/son.nguyen/.pyenv/versions/3.9.0/lib/python3.9/site-packages/numpy/core/include
export CXXFLAGS=-I/data05/home/son.nguyen/.pyenv/versions/3.9.0/lib/python3.9/site-packages/numpy/core/include
```
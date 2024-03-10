# How to build
https://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html

## Step 1: Compiling with Position Independent Code
```
g++ -c -Wall -Werror -fpic task1.cpp task2.cpp
```

## Step 2: Creating a shared library from an object file
```
cwd=$(pwd)
g++ -shared -std=c++14 -o libtask1.so task1.o
g++ -L$cwd -shared -std=c++14 -o libtask2.so task2.o -ltask1
```

## Step 3: Test
```python
from ctypes import cdll
task2 = cdll.LoadLibrary('./libtask2.so')
```

Linking with a shared library
```
cwd=$(pwd)
g++ -L$cwd -Wall -o test main.cpp -ltask2
./test
```
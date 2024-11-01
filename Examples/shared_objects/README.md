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

# Lessons Learned
## Lesson 1: gcc allows to create a solib with undefined symbols:
```Bash
g++ -L$(CWD) -shared -std=c++14 -o libtask2.so task2.o
```
In this case, `libtask2.so` has 2 undefined symbols: 
```Bash
readelf -sW libtask2.so | c++filt | grep UND | grep -v 'std::' | grep -v '_'

    60: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND Task1::Task1()
    61: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND Task1::perform()
```

## Lesson 2: linker needs to resolve all undefined symbols at link time
```Bash
g++ -L/data00/home/son.nguyen/git/ngocson2vn/cpp/Examples/shared_objects -Wall -o test_task3 test_task3.cpp -ltask3
/usr/bin/ld: /data00/home/son.nguyen/git/ngocson2vn/cpp/Examples/shared_objects/libtask3.so: undefined reference to `Task2::Task2()'
/usr/bin/ld: /data00/home/son.nguyen/git/ngocson2vn/cpp/Examples/shared_objects/libtask3.so: undefined reference to `Task2::perform()'
collect2: error: ld returned 1 exit status
make: *** [Makefile:12: task3] Error 1
```

Though, `test_task3.cpp` calls only `perform3()` which is defined in `task3.cpp`, at link time, linker still failed to link object modules because linker could not resolve all undefined symbols. Specifically, `libtask3.so` has 2 undefined symbols: `Task2::Task2()` and `Task2::perform()`

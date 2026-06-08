# Add extra include paths
Check default include paths:
```Bash
# C
clang -E -x c - -v < /dev/null

# C++
clang++ -E -x c++ - -v < /dev/null
```

Add extra include paths:
```Bash
export CPATH=/usr/include
export CPLUS_INCLUDE_PATH=/usr/include:/usr/include/c++/12
```

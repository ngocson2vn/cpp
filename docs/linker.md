# ld
```Bash
/usr/bin/ld: unrecognized option '--color-diagnostics'
```
Solution: tell the compiler to use lld:
```Bash
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=lld")
```
# Dynamic Linking with Shared Libraries
```
gcc -shared -fPIC -o libvector.so addvec.c multvec.c
```

# Error 1
./main: relocation error: ./main: symbol _ZN10tensorflow8GraphDefC1Ev version tensorflow not defined in file libtensorflow_cc.so.2 with link time reference

## What does it mean?

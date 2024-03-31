# Static Linking
gcc -c addvec.c multvec.c
ar rcs libvector.a addvec.o multvec.o

## Correct Command line Orders
```bash
gcc -static -o p2 main2.c ./libvector.a
```

## Wrong Command line Orders
```bash
gcc -static -o p2 ./libvector.a main2.c

/usr/bin/ld: /tmp/cc3B9tXQ.o: in function `main':
main2.c:(.text+0x19): undefined reference to `addvec'
collect2: error: ld returned 1 exit status
```


# Dynamic Linking with Shared Libraries
```bash
gcc -shared -fPIC -o libvector.so addvec.c multvec.c
```

# Error 1
./main: relocation error: ./main: symbol _ZN10tensorflow8GraphDefC1Ev version tensorflow not defined in file libtensorflow_cc.so.2 with link time reference

## What does it mean?

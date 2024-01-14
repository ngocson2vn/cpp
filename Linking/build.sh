#!/bin/bash

# Dynamic Linking
gcc -shared -fPIC -o libvector.so addvec.c multvec.c
gcc -o p2 main2.c libvector.so
#!/bin/bash

# Dynamic Linking
gcc -v -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -o libconcat.so concat.cc

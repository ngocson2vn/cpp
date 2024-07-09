#!/bin/bash

# Dynamic Linking
# g++ -o main -L. main.cc -lconcat
g++ -o main -D_GLIBCXX_USE_CXX11_ABI=0 -L. main.cc -lconcat

#!/bin/bash

# Build objects
# /usr/bin/g++ -g -std=c++14 -fno-elide-constructors -c *.cpp

# Build program
# /usr/bin/g++ -g -std=c++14 -fno-elide-constructors *.o -o main

make && echo

if [ "$?" == "0" ]; then
  ./main
fi

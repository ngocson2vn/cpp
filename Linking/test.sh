#!/bin/bash

g++ -c main.cpp foo.cpp bar.cpp

if [ "$?" == "0" ]; then
  g++ -o main_cpp main.o foo.o bar.o
fi

if [ "$?" == "0" ]; then
  ./main_cpp
fi
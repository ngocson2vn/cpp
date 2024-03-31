#!/bin/bash

g++ -c main.cpp student_factory.cpp dummy.cpp

if [ "$?" == "0" ]; then
  g++ -o main main.o student_factory.o dummy.o
fi

if [ "$?" == "0" ]; then
  ./main
fi

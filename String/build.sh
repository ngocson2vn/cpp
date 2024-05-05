#!/bin/bash

for src in $(ls *.cpp)
do
  echo "Building $src"
  p=$(echo $src | cut -d'.' -f1)
  g++ -o $p $src
done
#!/bin/bash

# Dynamic Linking
gcc -shared -fPIC -o libconcat.so concat.cc

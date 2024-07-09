#!/bin/bash

echo "Check libconcat.so:"
readelf -sW libconcat.so | grep concat | c++filt
echo
echo

echo "Check main:"
readelf -sW main | grep concat | c++filt
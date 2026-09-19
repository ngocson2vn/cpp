#!/bin/bash

for i in $(seq 1 10)
do
  echo "=================================================================="
  ./main1
  echo

  ./main2
  echo

  ./main3
  echo
done
#!/bin/bash

nsys profile \
  --force-overwrite=true \
  --trace=cuda,osrt \
  --sample=cpu \
  -o context_lock_trace1 \
  ./main1

echo
nsys profile \
  --force-overwrite=true \
  --trace=cuda,osrt \
  --sample=cpu \
  -o context_lock_trace2 \
  ./main2

echo
nsys profile \
  --force-overwrite=true \
  --trace=cuda,osrt \
  --sample=cpu \
  -o context_lock_trace3 \
  ./main3

#!/bin/bash

set -e

ROWS=3
COLS=259

if [[ -f ./main ]]; then
  ./main ${ROWS} ${COLS}
fi

if [[ -f ./test ]]; then
  ./test ${ROWS} ${COLS}
fi

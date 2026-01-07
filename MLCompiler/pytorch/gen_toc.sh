#!/bin/bash

set -e

target_md_file=$1

if [ "${target_md_file}" == "" ]; then
  echo "Please specify an existing md file path!"
  exit 1
elif [ ! -f ${target_md_file} ]; then
  echo "Please specify an existing md file path!"
  exit 1
fi

echo "MD file: ${target_md_file}"

python insert_toc.py ${target_md_file} --min-level 1 --max-level 4 --exclude "Table of Contents" --marker

#!/bin/bash

md_file_path="$1"

python insert_toc.py ${md_file_path} --min-level 1 --max-level 4 --exclude "Table of Contents" --marker

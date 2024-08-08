#!/bin/bash

set -e

echo "========================================"
echo "1. Current status"
git status
echo "========================================"
echo

git add -u

echo "========================================"
echo "2. Next status"
git status
echo "========================================"
echo

git commit -m "Update"
git push

echo "========================================"
echo "3. Final status"
git status
echo "========================================"
echo
#!/bin/bash

set -e

echo "================================================================================"
echo "1. Current status"
echo "================================================================================"
git status
echo "================================================================================"
echo

git add -u

echo "================================================================================"
echo "2. Next status"
echo "================================================================================"
git status
git commit -m "Update"
git push
echo "================================================================================"
echo

echo "================================================================================"
echo "3. Final status"
echo "================================================================================"
git status
echo "================================================================================"
echo
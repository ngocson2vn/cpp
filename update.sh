#!/bin/bash

set -e

git status
git add -u
git status
git commit -m "Update"
git push

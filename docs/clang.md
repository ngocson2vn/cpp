# Add extra include paths
Check default include paths:
```Bash
# C
clang -E -x c - -v < /dev/null

# C++
clang++ -E -x c++ - -v < /dev/null
```

Add extra include paths:
```Bash
export CPATH=/usr/include
export CPLUS_INCLUDE_PATH=/usr/include:/usr/include/c++/12
```

# Format
```Bash
COMMIT_HASH=730357c067a72b416145a019a2510c169d5e58c8
git --no-pager show --pretty="" --name-only ${COMMIT_HASH} | grep -E '\.h|\.cc' | xargs -I {} clang-format -i -style=file {}
```

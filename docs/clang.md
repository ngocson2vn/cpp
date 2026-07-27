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
COMMIT_HASH=9cc02903e9cf7a3d073c60a97033f0e4109a4863
git --no-pager show --pretty="" --name-only ${COMMIT_HASH} | grep -E '\.h|\.cc' | xargs -I {} clang-format -i -style=file {}

git --no-pager show --pretty="" --name-only HEAD | grep -E '\.h|\.cc' | xargs -I {} clang-format -i -style=file {}
```

# Overall Options
https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html#Overall-Options
```Bash
# Stop after the preprocessing stage; do not run the compiler proper. The output is in the form of preprocessed source code, which is sent to the standard output.
# Input files that don’t require preprocessing are ignored.
g++ -E

# Force noinline
g++ -fno-inline
```

# Include Path
```Bash
export CFLAGS=-I/data00/home/son.nguyen/.pyenv/versions/3.9.0/lib/python3.9/site-packages/numpy/_core/include
export CFLAGS=-I/data05/home/son.nguyen/.pyenv/versions/3.9.0/lib/python3.9/site-packages/numpy/core/include
```

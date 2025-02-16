# GDB Tips

## Define a custom command
vim ~/.gdbinit
```Python
define ni
  nexti
  disassemble $pc-40,$pc+30
end
```

## Print value at a memory address
```
(gdb) x/xg 0x7fff6a47d00a 
0x7fff6a47d00a: 0xb46000007ffedc00
```
x: Hex format  
g: Giant words (eight bytes).

## List functions
```C++
info functions xxx
```

# Pretty print
https://github.com/Microsoft/vscode-cpptools/issues/1414
https://sourceware.org/gdb/wiki/STLSupport

sudo apt install subversion -y
cd /data00/home/son.nguyen/workspace/tools/python
svn co svn://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/python

# Print string at address
x/s 0x7fe1dc3686c0

# info symbol
```C++
info symbol 0x7fe1dd9f8120
tensorflow::internal::LogMessageFatal::~LogMessageFatal()@got.plt in section .got.plt of ./lib/libtensorflow_framework.so.2
```

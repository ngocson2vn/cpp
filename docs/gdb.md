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
```Bash
info functions xxx
```
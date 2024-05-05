# Source node vs Sink node
In a Directed acyclic graph, a source node is a node (also known as a vertex) with no incoming connections from other nodes, while a sink node is a node without outgoing connections.

# Compiler
https://www.quora.com/What-are-the-best-resources-to-learn-about-C++-compiler

# GDB
## Print real type of an object
set print object on
whatis obj

## VSCode set gdb scheduler-locking step
1. Create gdb.conf in workspace folder
```bash
echo Adding hooks for stop to "set scheduler-locking step"\n
define hookpost-run
  set scheduler-locking step
end
define hookpost-attach
  set scheduler-locking step
end
```

2. Add this line to .vscode/launch.json
"miDebuggerArgs": "-x '${workspaceFolder}/gdb.conf'",

## Print vtable of an object
info vtbl obj

## Compound Statement
https://gcc.gnu.org/onlinedocs/gcc-13.2.0/gcc/Statement-Exprs.html
```c++
#include <iostream>

int check(int x) {
    if (x < 0) {
        return 0;
    }
    return x;
}

int main() {
    int x = 0;
    auto result = ({
        for (int i = 0; i < 10; i++) {
            x += i;
            std::cout << "x = " << x << std::endl;
        }
        check(x);
    });

    std::cout << "Result: " << result << std::endl;
}
```
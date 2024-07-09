# Source node vs Sink node
In a Directed acyclic graph, a source node is a node (also known as a vertex) with no incoming connections from other nodes, while a sink node is a node without outgoing connections.

# Compiler
https://www.quora.com/What-are-the-best-resources-to-learn-about-C++-compiler

# GDB
## Print real type of an object
set print object on
whatis obj

## Print vtable of an object
info vtbl obj

# std::map
For operator [ ], the type of the index is not the only difference from ordinary arrays. In addition,
you can’t have a wrong index. If you use a key as the index for which no element yet exists, a new
element gets inserted into the map automatically. The value of the new element is initialized by the
default constructor of its type. Thus, to use this feature, you can’t use a value type that has no default
constructor. Note that the fundamental data types provide a default constructor that initializes their
values to zero.

# How can an user-space application call a device driver function?
User-space app --> /dev/device file --> device driver functions

# Comma Operator
```C++
int a = (1, 2, 3);
```
This is the [comma operator](http://en.wikipedia.org/wiki/Comma_operator): evaluation of `a, b` first causes `a` to be evaluated, then `b`, and the result is that of `b`.

`int a = (1, 2, 3)`; first evaluates 1, then 2, finally 3, and uses that last 3 to initialise a. It is useless here, but it can be useful when the left operand of , has side effects (usually: when it's a function call).

# Squiggles
Red squiggly line:  
<img align="left" src="./images/red_squiggly_line.png" alt="Red Squiggly Line" width="50%" />
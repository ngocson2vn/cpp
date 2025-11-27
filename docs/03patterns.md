# Pointer to Implementation
### Core Idea
- The public class declaration in the header contains only:
  - Public interface methods.
  - A forward declaration of the `Impl` class.
  - A smart pointer (or raw pointer) to `Impl`.
- The `Impl` class lives in the `.cpp` and contains:
  - All private members.
  - Implementation of logic, including headers that would otherwise leak into the public header.

### Concrete Example: A Logger Class
[pattern_pointer_to_impl](../Examples/pattern_pointer_to_impl/)

# EBO (Empty Base Optimization) and ESO (Empty Subobject Optimization)
This pattern addresses a specific quirk in how C++ handles memory for empty classes.

**1. The Problem: The "Size 1" Rule**
In C++, no object is allowed to have a size of 0. Even an empty class with no data members must have a size of at least 1 byte.

Why? Because every object instance must have a unique memory address so that pointer arithmetic works correctly. If the size were 0, an array of Empty objects would all point to the same address.
```C++
class Empty {};

// Result: 1 (not 0)
std::cout << sizeof(Empty) << std::endl;
```

The Consequence: If you use Composition (holding an empty class as a member variable), you waste memory due to padding/alignment.
```C++
struct Derived {
    Empty e; // Takes 1 byte
    int i;   // Takes 4 bytes
    // + 3 bytes of padding to align 'i' to a 4-byte boundary
};

// Result: 8 bytes (1 + 3 padding + 4)
std::cout << sizeof(Derived) << std::endl;
```
You are using 8 bytes to store 4 bytes of data.

**2. The Solution: Empty Base Optimization (EBO)**
The C++ standard includes a special rule: If a class inherits from an empty base class, the compiler is allowed to treat the base class part as having size 0.

This means the compiler effectively "overlays" the empty base class at the same memory address as the first member of the derived class.
```C++
// Using Inheritance instead of Composition
struct OptimizedDerived : public Empty {
    int i; 
};

// Result: 4 bytes! 
// The 'Empty' base takes 0 space.
std::cout << sizeof(OptimizedDerived) << std::endl;
```

ESO stands for Empty Subobject Optimization.

It is the modern, generalized term for the Empty Base Optimization (EBO).

Why the name change?
EBO (Empty Base Optimization): Specifically referred to the "hack" where you had to use inheritance (a base class) to make an empty class take up 0 bytes.

ESO (Empty Subobject Optimization): Is the broader term. It acknowledges that in modern C++ (specifically C++20 and later), the compiler can optimize any empty subobject (including member variables), not just base classes.
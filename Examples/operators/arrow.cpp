#include <iostream>

struct A {
    void foo() { std::cout << "Hi" << std::endl; }
};

struct B {
    A a;

    A* operator->() {
        return &a;
    }
};

// So, when the compiler sees b->foo(), it performs the following steps:
// 1. Calls `b.operator->()`, resulting in a pointer to `a` (i.e., `&a`).
// 2. Then applies the arrow operator to the pointer `&a`, resulting in `(&a)->foo()`.
int main() {
    B b;
    // b->foo() is interpreted by the compiler as:
    // Step 1: b.operator->() -> returns &a
    // Step 2: (&a)->foo() -> calls foo() on a
    b->foo(); // Outputs: Hi
}

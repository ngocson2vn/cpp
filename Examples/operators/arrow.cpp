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

void print_counter(int counter) {
    std::cout << "Counter: " << counter << std::endl;
}

// So, when the compiler sees b->foo(), it performs the following steps:
// 1. Calls `b.operator->()`, resulting in a pointer to `a` (i.e., `&a`).
// 2. Then applies the arrow operator to the pointer `&a`, resulting in `(&a)->foo()`.
int main() {
    B b;
    // b->foo() is interpreted by the compiler as:
    // Step 1: b.operator->() -> returns &a
    // Step 2: (&a)->foo() -> calls foo() on a
    b->foo(); // Outputs: Hi

    int counter = 0;
    while (true) {
        print_counter(counter++);
        if (counter == 10) {
            break;
        }
    }
}

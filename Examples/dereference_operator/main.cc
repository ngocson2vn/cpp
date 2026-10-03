#include <iostream>
#include <string>

struct People {
  int id;
  std::string name;
};


template <typename T>
void display_type(T x) {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  // std::cout << "tmp = " << tmp << std::endl;
  std::string type = tmp.substr(4, tmp.size() - 5);
  std::cout << "T = " << type << std::endl;
}

/*
The default, built-in `*` operator **never** returns a reference type (`T&`). Instead, it yields an **lvalue expression** of type `T`.

While in everyday conversation C++ programmers often say "dereferencing a pointer returns a reference," the C++ Standard strictly separates the concept of *expressions* (what operators produce) from *references* (a language type used for variables and return values).

Here is exactly how the default built-in `*` operator behaves:
### 1. Expressions yield Lvalues, not Reference Types

If you have a pointer `int* p`, the expression `*p` does not have the type `int&`. According to the C++ Standard, the type of the expression `*p` is strictly `int`.

However, expressions in C++ also have a **value category**. The value category of `*p` is an **lvalue** (locator value), meaning it designates a specific location in memory. Because it is an lvalue, you can assign to it (`*p = 5`), and you can bind it to a reference variable (`int& ref = *p`), which is where the confusion usually stems from.

### 2. The `decltype` Illusion
Many developers believe `*` returns a reference because of how the `decltype` keyword behaves:
```cpp
int x = 5;
int* p = &x;
// decltype(*p) evaluates to int&
```

This happens because `decltype` has a special, hard-coded rule in the compiler: if you give it an expression that is an **lvalue** of type `T`, `decltype` artificially transforms the result into `T&`. The operator itself didn't return a reference; `decltype` just converted the lvalue into one.
*/
int main(int argc, char** argv) {
  People* ptr = new People{10, "Sony"};
  display_type(ptr);
  display_type(*ptr);
}

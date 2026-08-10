# How does linker handle duplicate definitions?
Given the following C++ header `point.h`:
```cpp
namespace Geometry {
class Point {
private:
  int x;
  int y;

public:
  Point(int x, int y) : x(x), y(y) {}

  // Hidden Friend 1: operator==
  // Defined inline inside the class. It is completely invisible to
  // standard name lookup in the Geometry namespace.
  friend bool operator==(const Point &lhs, const Point &rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
  }

  // Hidden Friend 2: operator<<
  // Often used for output streams so it can access private members.
  friend std::ostream &operator<<(std::ostream &os, const Point &p) {
    os << "(" << p.x << ", " << p.y << ")";
    return os;
  }
};
} // namespace Geometry
```

If this header `point.h` is included by two source files `source1.cc` and `source2.cc`, then both source files are linked with `main.cpp`. There will be 2 definitions of `bool operator==(const Point &lhs, const Point &rhs)`.

How does the linker handle the duplicate of `bool operator==(const Point &lhs, const Point &rhs)`?


The linker will handle this gracefully without throwing a "multiple definition" error.

Here is exactly how C++ and the linker manage this under the hood:

### 1. The Implicit `inline` Rule

In C++, any function—including a `friend` function—that is defined entirely within the body of a class is **implicitly marked as `inline`**.

Because you wrote the actual body of `operator==` (and `operator<<`) inside `class Point { ... }`, the C++ compiler treats both of them as inline functions, exactly as if you had typed `friend inline bool operator==(...)`.

### 2. The One Definition Rule (ODR) Exception

Normally, C++'s One Definition Rule dictates that a function can only be defined once in the entire program. If you break this rule with a regular non-inline function, the linker throws a duplicate symbol error.

However, **inline functions have a specific exception to the ODR**: they *must* be defined in every translation unit (source file) that uses them, and it is perfectly legal for the program to contain multiple identical definitions of an inline function across different object files.

### 3. How the Linker Resolves It

Since `source1.cc` and `source2.cc` both include `point.h`, both object files (`source1.o` and `source2.o`) will contain their own compiled version of `operator==`. When you link them together into `main`, the linker resolves the duplication using a mechanism known as **COMDAT folding** or **weak symbols**:

* **During Compilation:** The compiler marks the symbols for these inline functions as "weak" or places them in special "COMDAT" groups in the object files, rather than marking them as standard "strong" global symbols.
* **During Linking:** When the linker encounters multiple weak/COMDAT symbols with the exact same signature, it assumes they are identical (as required by the C++ standard). It simply **picks one instance to keep in the final executable and discards all the others**.

Because of this mechanism, your hidden friends are safely deduplicated, resulting in exactly one copy of the function in your final binary.

/*
## The "Hidden Friend" Rule

When you provide the function body inside the class, you are creating what C++
developers call a **hidden friend**. C++ rules dictate that this defines a
brand-new function in the immediate enclosing namespace (`Inner` in the example
below).

```cpp
namespace Outer {
    void myFunc(); // Outer declaration

    namespace Inner {
        class MyClass {
            // This DEFINES a new function: Inner::myFunc()
            // It completely ignores Outer::myFunc()
            friend void myFunc() {
                // ...
            }
        };
    }
}
```

Because `Inner::myFunc()` is defined inline inside the class, it remains
"hidden" from standard name lookup and can typically only be found via
Argument-Dependent Lookup (ADL) when an object of `MyClass` is passed to it.

## Why the "Hidden Friend" idiom usually omits the declaration

When C++ developers intentionally use the "hidden friend" idiom (especially for
things like overloaded `operator==` or `operator<<`), they specifically
**avoid** writing that prior declaration in the namespace.

By omitting the prior namespace declaration, they force the function to remain
invisible to standard lookup. This keeps the global or namespace scope clean,
reduces compile times (fewer overload candidates to check), and prevents
accidental implicit conversions—forcing the compiler to only find the function
via ADL when an object of `MyClass` is explicitly involved.

This is considered best practice because of the following reasons:

1. **Cleaner Namespaces:** If you declared `operator==` in the `Geometry`
namespace, the compiler would have to consider it every time you used `==` on
*any* types within or near that namespace, slowing down compilation. As a hidden
friend, it is only ever considered when a `Point` is actually involved.

2. **Prevents Accidental Conversions:** If `Point` had a constructor that took a
single `int` (e.g., `Point(int val)`), an external namespace `operator==` might
accidentally allow `if (p1 == 5)`. Hidden friends are much stricter because ADL
requires at least one argument to already be the exact class type before it even
looks for the hidden function.

3. **Encapsulation:** It keeps the interface tightly bound to the class itself.
You define the behavior of the class directly where the data is defined, without
polluting the surrounding namespace.
*/

#pragma once

#include <iostream>

namespace Geometry {
class Point {
private:
  int x;
  int y;

public:
  Point(int x, int y) : x(x), y(y) {}

  friend class Square;
  friend class Rectangular;

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

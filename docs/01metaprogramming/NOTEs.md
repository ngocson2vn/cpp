# declval
```C++
#include <utility>
```
`std::declval<T>()`: This is a template function (found in the <utility> header) that's used to produce a value of type `T` for the sole purpose of type deduction. <br/>
It's important to note that `std::declval` can only be used in unevaluated contexts (like `decltype` or `sizeof`). <br/>
You can't actually call declval and use its result in runtime code.

`decltype(*declval<T&>())` means:<br/>
1. Create a reference to an object of type `T` (using `declval<T&>()`). This reference exists only for the purpose of type deduction.
2. Dereference that reference (using `*`). This gives us an expression that represents the object of type `T` itself.
3. Determine the type of that expression (using `decltype`).

In essence, `decltype(*declval<T&>())` evaluates to the type `T`.

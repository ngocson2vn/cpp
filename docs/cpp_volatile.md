# volatile keyword
Source code: [../Examples/volatile_var/main.cpp](../Examples/volatile_var/main.cpp)
```cpp
volatile bool running = true;
```
Without `volatile` qualifier, the compiler will remove `while (running)` because of the following reasons:

1. Writing to the plain `bool running` from a signal handler is **undefined behaviour**.
2. Therefore the compiler may assume that the write never happens, i.e. `running` stays `true` forever.
3. Under that assumption, if the `while (running)` loop is ever entered it becomes an **infinite loop that has no observable side effects**.
4. An infinite loop with no observable side effects is **itself undefined behaviour** (C++ forward-progress rules).
5. A compiler is allowed to assume that undefined behaviour never occurs in any valid execution of the program.
6. Therefore the situation "the loop is entered and runs forever cannot happen in a valid program.
7. The only remaining possibility consistent with the program being free of UB is that the loop is never entered. Hence the compiler deletes it.

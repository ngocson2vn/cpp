#include <cstdio>

struct Widget {
    void write(FILE* os) const {
        fprintf(os, "This is Widget.\n");
    }
};

template<bool B, typename X = void>
struct enable_if {
};

template<typename X>
struct enable_if<true, X> {
    using type = X;
};

template <typename T>
struct uses_write {
    static const bool value = false;
};

template <>
struct uses_write<Widget> {
    static const bool value = true;
};

template <typename T>
constexpr bool uses_write_v = uses_write<T>::value;

// Template Argument Deduction:
// C++14 (ISO/IEC 14882:2014): Section [14.8.2] describes the process of template argument deduction. 
// Specifically, if a template parameter has a default argument, that argument is used when no argument is provided by the caller.
template <typename T, typename enable_if<uses_write_v<T>>::type* = nullptr>
void serialize(FILE* os, T const& value) {
    value.write(os);
}

int main(int argc, char** argv) {
    Widget w;

    // The compiler deduces `T` as Widget from the function call arguments.
    // The second parameter `typename enable_if<uses_write_v<T>>::type*` does not need to be explicitly deduced because it has a default value of `nullptr`.
    // Substitution happens after the deduction of `T`, ensuring the template is only instantiated if `uses_write_v<T>` is true (SFINAE applies).
    serialize(stdout, w);
}

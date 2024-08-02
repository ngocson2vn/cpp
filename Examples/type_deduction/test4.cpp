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

template <typename T, typename = typename enable_if<uses_write_v<T>>::type>
void serialize(FILE* os, T const& value) {
    value.write(os);
}

int main(int argc, char** argv) {
    Widget w;
    serialize(stdout, w);
}

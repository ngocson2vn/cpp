#include <cstdio>

struct Widget {
    void write(FILE* os) const {
        fprintf(os, "This is Widget.\n");
    }
};

struct Gadget {
    operator const char*() const {
        return "This is Gadget.";
    }

    const char* c_str() const {
        return "This is Gadget.";
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

template <typename T, typename enable_if<uses_write_v<T>>::type>
void serialize(FILE* os, T const& value) {
    value.write(os);
}

// template <typename T, typename enable_if<!uses_write_v<T>>::type>
// void serialize(FILE* os, T const& value) {
//     fprintf(os, "%s\n", value.c_str());
// }


int main(int argc, char** argv) {
    Widget w;
    serialize(stdout, w);

    // Gadget g;
    // serialize(stdout, g);
}

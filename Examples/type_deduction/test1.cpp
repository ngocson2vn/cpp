#include <cstdio>

template<bool B, typename X = int>
struct enable_if {
};

template<typename X>
struct enable_if<true, X> {
    using type = X;
};

int main(int argc, char** argv) {
    typename enable_if<true>::type v;
}

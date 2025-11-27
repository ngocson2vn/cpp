#include <iostream>
#include <type_traits>

template <class T, class = void>
struct has_dereference : std::false_type {};

template <class T>
struct has_dereference<T, std::void_t<decltype(*std::declval<T&>())>> : std::true_type {};

struct TypeA {
  int value;
};

struct TypeB {
  TypeB& operator*() {
    return *this; 
  }

  int value;
};

template <typename T>
void print_host_type(const char* name, bool lb = true) {
  const char* func_name = __PRETTY_FUNCTION__;
  // printf("%s\n", func_name);
  char* type = const_cast<char*>(func_name);
  char* ptr = type;
  int n = 0;
  while (*ptr != '=') {
    ptr++;
    n++;
  }

  type = type + (n + 2);
  ptr = type;

  n = 0;
  while (*ptr != ']') {
    ptr++;
    n++;
  }

  if (lb) {
    printf("\n%s: %.*s\n", name, n, type);
  } else {
    printf("%s: %.*s\n", name, n, type);
  }
}

int main() {
  // using DeducedTypeA = decltype(*std::declval<TypeA&>());
  // print_host_type<DeducedTypeA>();

  using DeducedTypeB = decltype(*std::declval<TypeB&>());
  print_host_type<DeducedTypeB>("DeducedTypeB", false);

  if constexpr(has_dereference<TypeA>::value) {
    std::cout << "TypeA has a dereference operator*()" << std::endl;
  } else {
    std::cout << "TypeA doesn't have a dereference operator*()" << std::endl;
  }

  if constexpr(has_dereference<TypeB>::value) {
    std::cout << "TypeB has a dereference operator*()" << std::endl;
  } else {
    std::cout << "TypeB doesn't have a dereference operator*()" << std::endl;
  }

  return 0;
}

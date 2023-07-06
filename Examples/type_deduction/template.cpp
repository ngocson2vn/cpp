#include <iostream>

template <typename T>
void f(T& param) {
  std::cout << "addr(param) = " << &param << std::endl;
  std::cout << "param = " << param << std::endl;
}

template <typename T, std::size_t N>
constexpr std::size_t arraySize(T (&)[N]) noexcept {
  return N;
}

int main() {
  const char name[] = "Son Nguyen";
  std::cout << "addr(name) = " << &name << std::endl;
  std::cout << "size(name) = " << arraySize(name) << std::endl;
  f(name);
}
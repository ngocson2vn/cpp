#include <cstdio>
#include <iostream>
#include <string>
#include <initializer_list>

template <typename... T>
void println(T... args) {
    auto arg_list = {args...};
    for (auto e : arg_list) {
        printf("%d\n", e);
    }
}

std::string BuildShapeString(const std::initializer_list<int>& dim_sizes) {
  std::string shape_str = "{";
  bool is_first = true;
  for (auto& e : dim_sizes) {
    if (is_first) {
      shape_str.append(std::to_string(e));
      is_first = false;
    } else {
      shape_str.append(", ");
      shape_str.append(std::to_string(e));
    }
  }
  shape_str.append("}");

  return shape_str;
}

int main() {
    println(1, 2, 3);
    auto dim_sizes = {10, 20, 30};
    std::cout << BuildShapeString(dim_sizes) << std::endl;
}

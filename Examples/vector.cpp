#include <iostream>
#include <vector>

#include "common.h"

void print_vector(const std::vector<int>& v) {
  for (auto& e : v) {
    std::cout << "Element: " << e << std::endl;
  }
}

int main() {
  std::vector<int> v;
  for (int i = 0; i < 10; i++) {
    v.push_back(i);
  }

  for (auto e : v) {
    std::cout << type_name<decltype(e)>() << std::endl;
    e = e * 2;
  }
  std::cout << std::endl;
  print_vector(v);
  std::cout << std::endl;

  for (auto& e : v) {
    std::cout << type_name<decltype(e)>() << std::endl;
    e = e * 2;
  }
  std::cout << std::endl;
  print_vector(v);
}
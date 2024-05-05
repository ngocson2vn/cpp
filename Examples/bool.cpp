#include <iostream>

int main() {
  bool b = false;
  size_t n = 10;
  b = true;
  bool r = b && n;
  std::cout << (r ? "true" : "false") << std::endl;
}
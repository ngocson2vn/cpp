#include <iostream>
#include <vector>

int main() {
  std::vector<int> v;
  for (int i = 0; i < 100; i++) {
    v.push_back(i);
  }

  for (auto& e : v) {
    std::cout << "Element: " << e << std::endl;
  }
}
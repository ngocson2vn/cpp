#include <iostream>
#include <string>

using FilterContainer = std::vector<std::function<bool(int)>>;
FilterContainer filters;

void addFilter() {
  static int divisor = 5;
  int x = 10;
  filters.emplace_back([x](int value) mutable {
    divisor++;
    x++;
    std::cout << "&divisor: " << &divisor << std::endl;
    std::cout << "divisor: " << divisor << std::endl;
    std::cout << "x: " << x << std::endl;
    return value % divisor == 0;
  });
  divisor++;
}

int main() {
  for (int i = 0; i < 3; i++) {
    addFilter();
  }

  for (auto& filter : filters) {
    bool ok = filter(10);
    std::cout << "ok: " << (ok ? "true" : "false") << std::endl;
    std::cout << std::endl;
  }
}

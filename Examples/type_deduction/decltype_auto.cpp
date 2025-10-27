#include <iostream>
#include <vector>
#include <utility>

template <typename T>
void printType() {
  std::string func_name = __PRETTY_FUNCTION__;
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  std::string type = tmp.substr(4, tmp.size() - 5);
  std::cout << type << std::endl;
}

template <typename Container, typename Index>
decltype(auto) f(Container&& c, Index i) {
  std::cout << "Container = "; printType<Container>();
  std::cout << "c has type = "; printType<decltype(c)>();
  return std::forward<Container>(c)[i];
}

int main(int argc, char** argv) {
  std::vector<int> v = {1, 2, 3};
  decltype(auto) ret1 = f(v, 0);
  std::cout << "ret1 has type = "; printType<decltype(ret1)>();
  std::cout << std::endl;

  decltype(auto) ret2 = f(std::vector<int>{1, 2, 3}, 0);
  std::cout << "ret2 has type = "; printType<decltype(ret2)>();
  std::cout << "ret2 has value = " << ret2 << std::endl;

  return 0;
}
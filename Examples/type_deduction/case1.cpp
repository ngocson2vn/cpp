#include <iostream>

template <typename T>
void printType() {
  std::string func_name = __PRETTY_FUNCTION__;
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  std::string type = tmp.substr(4, tmp.size() - 5);
  std::cout << type << std::endl;
}

template <typename T>
void f(T& param) {
  std::cout << "T = "; printType<T>();
  std::cout << "param has type = "; printType<decltype(param)>();
}

// int main(int argc, char** argv) {
//   f(std::string("a temporary string object"));
//   return 0;
// }

int main(int argc, char** argv) {
  std::string&& s = std::string("a temporary string object");
  std::cout << "s has address = " << &s << std::endl;
  std::cout << "s has type = "; printType<decltype(s)>();
  f(s);
  return 0;
}
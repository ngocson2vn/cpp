#include <iostream>
#include <string>
#include <vector>

#define FULL_NAME() \
do { \
  std::string func_name(__PRETTY_FUNCTION__); \
  std::cout << func_name << std::endl;  \
} while(0)

template <typename T>
std::string get_type() {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  std::string type = tmp.substr(4, tmp.size() - 5);
  return type;
}

template <typename T, typename... Args>
void printArgs(T arg0, Args&&... args) {
  FULL_NAME();
  std::cout << arg0 << "\n\n";
  if constexpr(sizeof...(args) > 0) {
    printArgs(args...);
  }
}

int main(int argc, char** argv) {
  std::string hello = "Hello";
  std::string world = "world";
  
  printArgs<std::string>(hello, world, 2026);
  // The compiler will deduce types for Args parameter
}

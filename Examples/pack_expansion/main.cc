#include <iostream>
#include <string>

template<typename T>
void printType() {
  std::string type = __PRETTY_FUNCTION__;
  std::cout << type << std::endl;
}

template <typename... Args>
void addOperations() {
  // This initializer_list argument pack expansion is essentially equal to
  // using a fold expression with a comma operator. Clang however, refuses
  // to compile a fold expression with a depth of more than 256 by default.
  // There seem to be no such limitations for initializer_list.
  (void)std::initializer_list<int>{0, (printType<Args>(), 0)...};
}

int main(int argc, char** argv) {
  addOperations<char, int, float, std::string>();
}

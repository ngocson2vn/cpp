#include <iostream>
#include <string>

template<typename T>
void printType() {
  std::string type = __PRETTY_FUNCTION__;
  std::cout << type << std::endl;
}

template<typename... Types>
void printTypes() {
  (printType<Types>(), ...);
}

int main(int argc, char** argv) {
  printTypes<char, int, float, std::string>();
}

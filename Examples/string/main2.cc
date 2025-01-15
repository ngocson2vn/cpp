#include <iostream>
#include <string>

template <typename T>
class TD;

template <size_t N>
void print1(const char (&str)[N]) {
  std::cout << "print1():" << std::endl;
  std::cout << "  String: " << str << std::endl;
  std::cout << "  Length: " << N - 1 << std::endl; // Ignore the last character which is a null character
}

void print2(const char* str) {
  std::cout << "print2():" << std::endl;
  std::cout << "  String: " << str << std::endl;
  std::cout << "  Length: " << std::string(str).size() << std::endl;
}

int main(int argc, char** argv) {
  // TD<decltype("Hello")> td;
  print1("Hello");
  std::cout << std::endl;
  print2("Hello");
}

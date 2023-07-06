#include <iostream>
#include <typeinfo>

using Buffer = char[8];

struct Input {
  int a;
  int b;

  void print() {
    std::cout << "Input.a = " << a << std::endl;
    std::cout << "Input.b = " << b << std::endl;
  }
};

int main() {
  int buffer[2] = {5, 7};
  std::cout << "The address of buffer: " << buffer << std::endl;

  auto input = reinterpret_cast<Input*>(buffer);
  std::cout << "The address of input:  " << input << std::endl;
  input->print();
}

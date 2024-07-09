#include "Derived.h"

template <typename T>
class TD;

int main() {
  Derived myDerived;
  std::cout << "myDerived: " << &myDerived << std::endl;
  myDerived.setIntValue(100);
  myDerived.showIntValue();
}
#include <iostream>
#include "foo.h"

void SetDefaultFooName();

int main() {
  std::cout << "This is main function" << std::endl;
  SetDefaultFooName();
  std::cout << "Foo object with default name: " << foo.getFooName() << std::endl;
  foo.setFooName("FooFooFoo");
  std::cout << "Foo object with updated name: " << foo.getFooName() << std::endl;
}

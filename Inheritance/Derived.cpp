#include "Derived.h"
#include <iostream>


void Derived::someMethod() {
  std::cout << "This is Derived's version of someMethod()." << std::endl;
}

void Derived::someOtherMethod() {
  std::cout << "This is Derived's version of someOtherMethod()." << std::endl;
}

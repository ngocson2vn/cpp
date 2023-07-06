#include "common.h"
#include <iostream>

int main() {
  const int theAnswer = 42;
  auto x = theAnswer;
  auto y = &theAnswer;

  std::cout << type_name<decltype(x)>() << std::endl;
  std::cout << type_name<decltype(y)>() << std::endl;
}
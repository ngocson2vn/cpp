#include <iostream>
#include "foo.h"

Foo::Foo() {
  std::cout << "This is default Foo constructor\n\n" << std::endl;
}

void Foo::setFooName(const char* name) {
  name_ = const_cast<char*>(name);
}

const char* Foo::getFooName() {
  return const_cast<const char*>(name_);
}

Foo foo;
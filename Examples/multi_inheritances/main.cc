#include <iostream>

class A {
 public:
  int i;
};

class B {
 public:
  int j;
};

class C: public A, public B {
 public:
  int k;
};

int main() {
  C c;
  C* cptr = &c;
  std::cout << "cptr = " << cptr << std::endl;

  A* aptr = &c;
  std::cout << "aptr = " << aptr << std::endl;

  B* bptr = &c;
  std::cout << "bptr = " << bptr << std::endl;

  return 0;
}
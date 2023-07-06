#include "Derived.h"

int main() {
  Derived myDerived;
  Base& ref = myDerived;
  ref.someMethod();
}
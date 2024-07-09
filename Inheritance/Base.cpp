#include "Base.h"
#include <iostream>

void Base::someMethod() {
  std::cout << "This is Base's version of someMethod()." << std::endl;
}

void Base::setIntValue(int value) {
  std::cout << "this: " << this << std::endl;
  this->mPrivateInt = value;
}

void Base::showIntValue() {
  std::cout << "this: " << this << std::endl;
  std::cout << "mPrivateInt: " << this->mPrivateInt << std::endl;
  increaseIntValue();
  std::cout << "mPrivateInt: " << this->mPrivateInt << std::endl;
}

void Base::increaseIntValue() {
  std::cout << "this: " << this << std::endl;
  this->mPrivateInt++;
}
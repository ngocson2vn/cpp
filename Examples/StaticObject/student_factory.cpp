#include <iostream>
#include <string>
#include <memory>
#include <typeinfo>
#include <unordered_map>
#include "student_factory.h"

typedef std::unordered_map<std::string, std::shared_ptr<StudentFactory>> StudentFactories;
StudentFactories* factories() {
  static StudentFactories* factories_ = new StudentFactories;
  return factories_;
}

void StudentFactory::Register(const std::string& student_type, std::shared_ptr<StudentFactory> factory) {
  auto ret = factories()->insert({student_type, factory});
  if (ret.second) {
    std::cout << "Successfully registered " << typeid(*factory.get()).name() << std::endl;
  }
}

std::shared_ptr<StudentFactory> StudentFactory::GetFactory(const std::string& student_type) {
  auto ret = factories()->find(student_type);
  if (ret != factories()->end()) {
    return ret->second;
  }

  return nullptr;
}

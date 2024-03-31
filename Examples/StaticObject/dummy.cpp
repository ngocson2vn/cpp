#include <iostream>
#include <memory>

class Dummy {
  public:
    Dummy() {
      std::cout << "Default Dummy Constructor" << std::endl;
    }

    Dummy(const char* name) {
      std::cout << "Creating " << name << std::endl;
      name_ = const_cast<char*>(name);
      std::cout << "name_ = " << name_ << std::endl;
    }

    const char* getName() {
      return const_cast<const char*>(name_);
    }

    char* name_;
};

Dummy d("Dummy");

namespace dummy {
  Dummy d2("Dummy2");
}
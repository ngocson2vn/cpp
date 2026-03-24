// Ref: https://en.cppreference.com/w/cpp/language/adl

#include <iostream>

template <typename T>
void display_type() {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  std::string type = "type" + tmp.substr(1, tmp.size() - 2);
  std::cout << type << std::endl;
}

class TypeStorage {
 private:
  const void* storage;
};

class Type {
 public:
  using ImplType = TypeStorage;
};

class IntegerTypeStorage {
 private:
  const void* storage;
};

template <typename ConcreteT, typename BaseT, typename StorageT>
class TypeBase : public BaseT {
 public:
  using ImplType = StorageT;
 protected:
  int width;
};

class IntegerType : public TypeBase<IntegerType, Type, IntegerTypeStorage> {
  IntegerType(int w) {
    this->width = w;
  }
};

template <typename T>
void verify() {
  display_type<typename T::ImplType>();
}

int main(int argc, char** argv) {
  verify<IntegerType>();
}

// IntegerType::ImplType will be IntegerTypeStorage
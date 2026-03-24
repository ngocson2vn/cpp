#include <iostream>

template <typename T>
std::string get_type() {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string type = func_name.substr(func_name.find_first_of("["));
  return type;
}

class TypeStorage {

};

class IntegerTypeStorage : public TypeStorage {
 public:
  IntegerTypeStorage(int w) : width(w) {

  }
 private:
  const int width;
};

class Type {
 public:
  Type() {
    std::cout << "Default Type() ctor " << get_type<decltype(this)>() << std::endl;
  }

  Type(TypeStorage* impl) : impl(impl) {
    std::cout << "Parameterized Type(ImplType* impl) ctor " << get_type<decltype(this)>() << std::endl;
  }

 private:
  TypeStorage* impl;
};


class TypeBase : public Type {
 public:
  using Type::Type;

  TypeBase() {
    std::cout << "Default TypeBase() ctor " << get_type<decltype(this)>() << std::endl;
  }
};

class IntegerType : public TypeBase {
 public:
  using TypeBase::TypeBase;

  IntegerType() {
    std::cout << "Default IntegerType() ctor " << get_type<decltype(this)>() << std::endl;
  }

  static IntegerType get(int w) {
    return new IntegerTypeStorage(w);
  }
};

int main(int argc, char** argv) {
  auto t = IntegerType::get(32);
}

// Output:
// Parameterized Type(ImplType* impl) ctor [T = Type *]
// Ref: https://en.cppreference.com/w/cpp/language/adl

#include <iostream>

class Currency {
 public:
  Currency(const std::string& name) : name_(name) {

  }

 private:
  std::string name_;
};

void dummy_func(const char* str) {
  std::cout << str << std::endl;
}

template <typename T>
void display_type() {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  std::string type = "type" + tmp.substr(1, tmp.size() - 2);
  std::cout << type << std::endl;
}

int main(int argc, char** argv) {
  char name[] = "Son Nguyen";
  // TD<decltype(name)> td;
  std::cout << name << std::endl;

  name[0] = 'T';
  std::cout << name << std::endl;

  const char name2[] = "Son Nguyen";
  display_type<decltype(name2)>();
}

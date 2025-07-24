#include <iostream>
#include <string>

#define FULL_NAME() \
do { \
  std::string func_name(__PRETTY_FUNCTION__); \
  std::string name = func_name.substr(0, func_name.find_first_of("["));  \
  std::cout << name << std::endl;  \
} while(0)

template <typename T>
std::string get_type() {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  std::string type = tmp.substr(4, tmp.size() - 5);
  return type;
}

template <typename T>
struct base_parser {
  void init() {
    FULL_NAME();
  }
};

template <>
struct base_parser<int> {
  // static constexpr int init = 0;
  void init() {
    FULL_NAME();
  }
};

template <typename T>
struct parser : base_parser<T> {
  void parse() {
    // At this point, init is a non-dependent name because
    //   1. The base_parser can be specialized and init can become anything or even missing
    //   2. So the compiler won't consider the base class template at the first phase name lookup
    // Because init is a non-dependent name, it must exist at the first phase name lookup
    // In this case, init doesn't exist, so the compiler will spout "error: use of undeclared identifier 'init'".
    // init();

    // This following fix turns init into a dependent name.
    // But at the second phase of name lookup,
    //   1. The compiler will spout "error: no member named 'init' in 'parser<int>'" if the specialized base_parser<int> removes init.
    //   2. The compiler will spout "error: called object type 'int' is not a function or function pointer" if init is a static data member.
    this->init();

    FULL_NAME();
  }
};

int main(int argc, char** argv) {
  parser<int> p;
  p.parse();
}

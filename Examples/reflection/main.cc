#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <functional>

#define quote(x) #x

typedef void (*func_ptr_t)();

class MyClass {
 public:
  static void Compute() {
    std::cout << "Compute\n";
  }
};


int main(int argc, char** argv) {
  MyClass m;
  int status;
  char* name1 = abi::__cxa_demangle(typeid(m).name(), 0, 0, &status);
  std::cout<< name1 << "\t" << quote(m) << "\n";

  char* name2 = abi::__cxa_demangle(typeid(MyClass::Compute).name(), 0, 0, &status);
  std::cout<< name2 << "\t" << quote(m.Compute) << "\n";

  free(name1);

  return 0;
}

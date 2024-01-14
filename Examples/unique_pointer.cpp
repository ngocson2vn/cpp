#include <iostream>
#include <memory>

class ClassFoo {
  public:
    ClassFoo() {
      std::cout << "Construct ClassFoo" << std::endl;
      this->id = 5;
    }

    void PrintFooID() {
      std::cout << "Foo ID: " << this->id << std::endl;
    }
  
  private:
    int id;
};

class ClassBar {
  public:
    ClassBar() {
      std::cout << "Construct ClassBar" << std::endl;
      this->id = 10;
    }

    void PrintBarID() {
      std::cout << "Bar ID: " << this->id << std::endl;
    }
  
  private:
    int id;
};

struct FunctionInfo {
  std::unique_ptr<ClassFoo> foo_ptr;
  std::unique_ptr<ClassBar> bar_ptr;
};

int main() {
  std::unique_ptr<FunctionInfo> func_info(new FunctionInfo);

  func_info->foo_ptr->PrintFooID();
  ClassFoo* raw_foo_ptr = func_info->foo_ptr.get();
  raw_foo_ptr->PrintFooID();

  func_info->bar_ptr->PrintBarID();
  ClassBar* raw_bar_ptr = func_info->bar_ptr.get();
  raw_bar_ptr->PrintBarID();

  return 0;
}

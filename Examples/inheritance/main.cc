#include <iostream>
#include <string>
#include <cstring>

class Base {
 public:
  void init() {
    std::string func_name(__PRETTY_FUNCTION__);
    std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
    std::string type = "type = " + tmp.substr(0, tmp.size());
    std::cout << type << std::endl;

    char init_data[] = "Initial data";
    data_ = new char[128];
    std::memcpy(data_, init_data, sizeof(init_data));
  }

  virtual void print_data() {
    std::string func_name(__PRETTY_FUNCTION__);
    std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
    std::string type = "type = " + tmp.substr(0, tmp.size());
    std::cout << type << std::endl;

    std::cout << "data: " << data_ << std::endl;
  }

 private:
  char* data_;
};

class Derived : public Base {
 public:
  // virtual void print_data_derived() {
  //   std::cout << "data: " << data_ << std::endl;
  // }
  void do_something() {
    std::cout << "Derived::do_something()" << std::endl;
  }
};

template <typename T>
void display_type() {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  std::string type = "type" + tmp.substr(1, tmp.size() - 2);
  std::cout << type << std::endl;
}

int main(int argc, char** argv) {
  // Derived d;
  // d.print_data();
  Base* d = new Derived();
  
  d->init();
  d->print_data();
}

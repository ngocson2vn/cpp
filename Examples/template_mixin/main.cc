#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <functional>

template <typename Derived, template <typename> class... Mixins>
class Base : public Mixins<Derived>... {
 public:
  void process() {
    static_cast<Derived*>(this)->process_impl();
  }

 // Ensure that Base must be inherited
 protected:
  Base() = default;
  ~Base() = default;
};

template <typename Derived>
class TypeName {
 public:
  std::string getTypeName() {
    return type_name_;
  }

 // Ensure that TypeName must be inherited
 protected:
  TypeName() {
    std::string func_name(__PRETTY_FUNCTION__);
    std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
    type_name_ = tmp.substr(10, tmp.size() - 11);
    std::cout << "Creating an instance of type = " << type_name_ << std::endl;
  }

  ~TypeName() = default;

 private:
  std::string type_name_;
};

template <typename Derived>
class Printer {
 // Ensure that Printer must be inherited
 protected:
  Printer() = default;
  ~Printer() = default;

  // Default implementations
  std::string dump() {
    return "<<DEFAULT>>";
  }

 public:
  void print() {
    auto derivedPtr = static_cast<Derived*>(this);
    std::cout << "==========================================" << std::endl;
    std::cout << "Type = " << derivedPtr->getTypeName() << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << derivedPtr->dump() << std::endl;
  }
};

class StringDataProcessor : public Base<StringDataProcessor, TypeName, Printer> {
 public:
  StringDataProcessor() = default;
  StringDataProcessor(const std::string& data) : data_(data) {};

  void process_impl() {
    std::cout << "Processing std::string data ..." << std::endl;
  }

  std::string dump() {
    return data_;
  }

 private:
  std::string data_;
};

class VectorDataProcessor : public Base<VectorDataProcessor, TypeName, Printer> {
 public:
  VectorDataProcessor() = default;
  VectorDataProcessor(const std::vector<int>& data) : data_(data) {};

  void process_impl() {
    std::cout << "Processing std::vector<int> data ..." << std::endl;
  }

  std::string dump() {
    std::string ret;
    std::stringstream ss(ret);
    ss << "data = [";
    for (int i = 0; i < data_.size() - 1; i++) {
      ss << data_[i] << ", ";
    }
    ss << data_.back() << "]";

    return ss.str();
  }

 private:
  std::vector<int> data_;
};

int main() {
  StringDataProcessor sp("This is a sample string");
  sp.print();
  sp.process();
  std::cout << sp.getTypeName() << std::endl;

  std::cout << std::endl;

  VectorDataProcessor vp({0, 1, 2, 3});
  vp.print();
  vp.process();
  std::cout << vp.getTypeName() << std::endl;

  return 0;
}

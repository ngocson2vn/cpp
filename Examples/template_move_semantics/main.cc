#include <iostream>
#include <string>

namespace status {

template <typename ValueT>
class Result {
 public:
  Result(ValueT&& val, const std::string& err = "") : value_(val), error_(err) {}

  // Enable move semantics
  Result(Result&& other) = default;
  Result& operator=(Result&& other) = default;

  // Disable copy constructor and copy assignment operator
  Result(const Result& other) = delete;
  Result& operator=(const Result& other) = delete;

  bool ok() const {
    return error_.empty();
  }

  const ValueT& value() const {
    return value_;
  }

  const std::string& error_message() const {
    return error_;
  }

 private:
  ValueT value_;
  std::string error_;
};

}

int main() {
  status::Result<std::string> res1("Test result 1");
  std::cout << res1.value() << "\n";

  status::Result<std::string> res2("Test result 2");
  std::cout << res2.value() << "\n";
  std::cout << "======================" << std::endl << std::endl;

  // error: overload resolution selected deleted operator '='
  // res2 = res1;
  
  // error: call to deleted constructor of 'status::Result<std::string>'
  // status::Result<std::string> res3 = res2;

  res2 = std::move(res1);
  std::cout << res2.value() << "\n";

  status::Result<std::string> res3 = std::move(res2);
  std::cout << res3.value() << "\n";
}
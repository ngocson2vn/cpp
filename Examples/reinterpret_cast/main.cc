#include <iostream>

class DenseValue {
 public:
  DenseValue(float v) : value_(v) {}

  explicit operator float() const {
    return value_;
  }

  float value() const {
    return value_;
  }

 private:
  float value_;
};

void printValue(float value) {
  std::cout << "Value: " << value << std::endl;
}

int main(int argc, char** argv) {
  float x = 0.5;
  std::cout << "x: " << x << std::endl;

  void* xptr = &x;

  float v = *reinterpret_cast<decltype(&x)>(xptr);
  std::cout << "v: " << v << std::endl;

  return 0;
}
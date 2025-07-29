#include <iostream>

class DenseValue {
 public:
  DenseValue(float v) : value_(v) {}

  explicit operator float() const {
    return value_;
  }

 private:
  float value_;
};

void printValue(float value) {
  std::cout << "Value: " << value << std::endl;
}

int main(int argc, char** argv) {
  DenseValue v(0.5);
  // printValue(v); // will not compile
  printValue(static_cast<float>(v)); // OK

  return 0;
}
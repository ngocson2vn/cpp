// Ref: https://en.cppreference.com/w/cpp/language/adl

#include <iostream>

constexpr static int kDataSize = 10;

class Foo {
 public:
  void print() {
    std::cout << "data:";
    for (int i = 0; i < kDataSize; i++) {
       std::cout << " " << data[i];
    }
    std::cout << std::endl;
  }

 private:
  int data[kDataSize];

  friend void init(Foo& foo) {
    for (int i = 0; i < kDataSize; i++) {
      foo.data[i] = 100;
    }
  }
};

int main(int argc, char** argv) {
  Foo foo;
  init(foo);
  foo.print();
}

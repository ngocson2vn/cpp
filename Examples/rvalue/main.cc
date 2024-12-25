/*
clang++ -c -std=c++14 -g -O0 main.cc
main.cc:24:3: error: no matching function for call to 'f'
  f(static_cast<Foo&&>(foo));
  ^
main.cc:11:6: note: candidate function not viable: expects an l-value for 1st argument
void f(Foo& foo) {
     ^
*/

#include <iostream>

struct Foo {
  void print() {
    std::cout << "This is Foo with address " << this << std::endl;
  }
};

void f(Foo& foo) {
  foo.print();
}

// void f(Foo&& foo) {
//   foo.print();
// }

// template <typename T>
// void type(T arg);

int main(int argc, char** argv) {
  Foo foo;
  f(static_cast<Foo&&>(foo));
}

// Ref: https://en.cppreference.com/w/cpp/language/adl

#include <iostream>

template <typename T>
struct TD;

struct X {
  int data;
  void print() {
    std::cout << "data: " << data << std::endl;
  }
};

struct Foo {
  int data;
  constexpr X with(int& data) const {
    return X{data};
  }
};

struct Bar : Foo {
  using Traits = Foo;

  template <typename Arg>
  X with(Arg&& data) const {
    // TD<decltype(data)> td;
    // TD<decltype(static_cast<Arg&>(data))> td;
    auto x = Traits::with(static_cast<Arg&>(data));
    return x;
  }
};

int main(int argc, char** argv) {
  Bar bar;
  int data = 100;
  auto x = bar.with(data);
  // auto x = bar.with(10);
  x.print();
}

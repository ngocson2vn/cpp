#include <iostream>
#include <memory>

class Base {
  public:
    Base() {
      std::cout << "Default Base Constructor\n";
    }

    virtual void print() {
      std::cout << "This is a Base instance\n";
    }
};

class Derived : public Base {
  public:
    Derived() {
      std::cout << "Default Derived Constructor\n";
    }

    virtual void print() override {
      std::cout << "This is a Derived instance\n";
    }
};

void playWithUniquePtr(std::unique_ptr<Base> base) {
  base->print();
}

int main() {
  playWithUniquePtr(std::make_unique<Derived>());
}

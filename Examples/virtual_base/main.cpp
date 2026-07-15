#include <iostream>

class Animal {
 public:
  void sleep() {
    std::cout << "Sleep\n";
  }
};

class Dog : public Animal {
 public:
  void run(int d) {
    std::cout << "Run " << d << " m\n";
  }
};

class Bird : public Animal {
  void fly(int d) {
    std::cout << "Fly " << d << " m\n";
  }
};

class DogBird : public Dog, public Bird {
 public:
  using Dog::sleep;
};

int main() {
  DogBird db;
  db.sleep();
}

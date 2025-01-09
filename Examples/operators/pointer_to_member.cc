#include <iostream>
using namespace std;

class MyClass {
 public:
  MyClass(int v) : value_(v) {}

  void display() {
    cout << "[display] Value: " << value_ << endl;
  }

  void print() {
    cout << "[print] Value: " << value_ << endl;
  }

 private:
  int value_;
};

template <typename T>
class TD;

int main() {
  MyClass obj(42);
  MyClass* objPtr = &obj;

  // Pointer to a member function
  void (MyClass::* funcPtr)() = &MyClass::display;
  // TD<decltype(funcPtr)> td; // TD<void (MyClass::*)()>

  // Call the member function using the ->* operator
  (objPtr->*funcPtr)(); // Output: Value: 42

  // Point it to another member function
  funcPtr = &MyClass::print;
  (objPtr->*funcPtr)();

  return 0;
}

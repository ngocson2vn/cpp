#include <iostream>

class Human;

class Student {

};

int main(int argc, char** argv) {
  // auto s = sizeof(void); // error: invalid application of 'sizeof' to an incomplete type 'void'
  // std::cout << "sizeof(void) = " << s << std::endl;

  // auto s2 = sizeof(Human); // error: invalid application of 'sizeof' to an incomplete type 'Human'
  // std::cout << "sizeof(Human) = " << s2 << std::endl;

  auto s3 = sizeof(Student);
  std::cout << "sizeof(Student) = " << s3 << std::endl;
}

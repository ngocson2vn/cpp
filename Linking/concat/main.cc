#include <iostream>

#include "concat.h"

int main(int argc, char** argv) {
  std::string s1 = "Son";
  std::string s2 = "Nguyen";
  std::string name = concat(s1, s2);
  std::cout << "Name: " << name << std::endl;
}

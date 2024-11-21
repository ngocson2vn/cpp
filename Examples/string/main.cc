#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string* s = new std::string("test clear method");
  std::cout << "BEFORE Capacity: " << s->capacity() << std::endl;
  s->clear();
  std::cout << "AFTER Capacity: " << s->capacity() << std::endl;
  // std::string s = "a test string a test string a test string a test string a test string a test string";
  // std::cout << "BEFORE Capacity: " << s.capacity() << std::endl;
  // s.clear();
  // s.shrink_to_fit();
  // std::cout << "AFTER Capacity: " << s.capacity() << std::endl;
}

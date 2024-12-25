// Ref: https://en.cppreference.com/w/cpp/language/adl

#include <iostream>
#include <string>
#include <bitset>

template <typename T>
struct TD;

struct X {
  int data;
  void print() {
    std::cout << "data: " << data << std::endl;
  }
};

int main(int argc, char** argv) {
  //
  // X object
  //
  X&& x = {10};
  printf("[BF] Address of x: %p, value of x: %d\n", &x, x.data);
  printf("[BF] Bits of x:");
  char* px = reinterpret_cast<char*>(&x);
  for (int i = 0; i < sizeof(x); i++) {
    printf(" %s", std::bitset<8>(px[i]).to_string().c_str());
  }
  printf("\n");
  
  x = {20};
  printf("[AT] Address of x: %p, value of x: %d\n", &x, x.data);
  printf("[AT] Bits of x:");
  px = reinterpret_cast<char*>(&x);
  for (int i = 0; i < sizeof(x); i++) {
    printf(" %s", std::bitset<8>(px[i]).to_string().c_str());
  }
  printf("\n\n");

  //
  // std::string object
  //
  std::string&& s = std::string("hello hello hello hello hello hello hello hello hello hello hello hello hello");
  printf("[BF] Address of s: %p, value of s: %p\n", &s, s.data());
  printf("[BF] Bits of s:");
  char* ps = reinterpret_cast<char*>(&s);
  for (int i = 0; i < sizeof(s); i++) {
    printf(" %s", std::bitset<8>(ps[i]).to_string().c_str());
  }
  printf("\n");

  s = std::string("world world world world world world world world world world world world world world world world world world world world world world world world world world");

  // The memory address of `s` will not be changed.
  printf("[AT] Address of s: %p, value of s: %p\n", &s, s.data());

  // The content of `s` will be changed.
  printf("[AT] Bits of s:");
  ps = reinterpret_cast<char*>(&s);
  for (int i = 0; i < sizeof(s); i++) {
    printf(" %s", std::bitset<8>(ps[i]).to_string().c_str());
  }
  printf("\n");
}

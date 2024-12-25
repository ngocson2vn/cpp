#include <cstdio>
#include <iostream>
#include <bitset>

int main(int argc, char** argv) {
  int32_t val = 1;
  int32_t mask = 1;
  for (int i = 0; i < 3; i++) {
    mask = mask << 8;
    val = val | mask;
    std::cout << "i = " << i << ", val = " << std::bitset<32>(val) << std::endl;
  }
}
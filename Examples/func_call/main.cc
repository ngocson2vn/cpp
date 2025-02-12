#include <cstdio>

void f1(int i, const char* str) {
  printf("Character at index %d: %c\n", i, str[i]);
}

void f2(int i1, int i2) {
  printf("i1 = %d, i2 = %d\n", i1, i2);
}

int main(int argc, char** argv) {
  f1(1, "Hello");
  f2(0x1, 0x2);
}

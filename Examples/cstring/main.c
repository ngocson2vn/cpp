#include <stdio.h>

int main(int argc, char** argv) {
  char* str = "Hello";
  printf("%d\n", *str);
  printf("%d\n", *str + 1);
  printf("%c\n", *str + 1);
}

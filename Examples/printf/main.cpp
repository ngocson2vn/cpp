#include <cstdio>

int main(int argc, char** argv) {
  const char* str = "This is a test message";
  int n = 14;

  // Print only first n characters
  printf("%.*s\n", n, str);

  float v = 1280.0 / 6;
  printf("%.2f\n", v);
}

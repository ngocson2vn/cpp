#include <iostream>

#define M1(name) #name

#define M2(name) #name

#define M(name) M1(name1) #name M2(name2)

int main(int argc, char** argv) {
  printf("%s\n", M(x));
}
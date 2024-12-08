#include <iostream>
#include <string>

#define varn(n) var##n;
#define varidx(n) varn(n)
#define var varidx(__COUNTER__)

std::string var; // varidx(__COUNTER__) --> varidx(0) --> varn(0) --> var##0 --> var0
std::string var; // varidx(__COUNTER__) --> varidx(1) --> varn(1) --> var##1 --> var1

int main(int argc, char** argv) {
  var0 = "var0";
  std::cout << var0 << std::endl;

  var1 = "var1";
  std::cout << var1 << std::endl;
}

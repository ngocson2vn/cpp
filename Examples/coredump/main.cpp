#include <iostream>
#include <ctime>

int main() {
  int64_t ts = (int64_t)std::time(0);
  std::string out = std::to_string(ts);
  std::cout << out << std::endl;
}

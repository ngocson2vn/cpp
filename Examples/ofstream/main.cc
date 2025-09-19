#include <iostream>
#include <fstream>

int main() {
  std::ofstream ofs("./output.txt", std::ios::binary);
  std::string input_str = "This is a test message";
  ofs.write(input_str.c_str(), input_str.size());
  ofs.close();
}
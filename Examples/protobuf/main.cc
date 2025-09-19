#include <iostream>
#include <fstream>

int main() {
  std::ofstream ofs("./output.txt", std::ios::binary);
  std::string input_str;
  
  // Suppose that input is a protobuf message
  // input->SerializeToString(&input_str);

  ofs.write(input_str.c_str(), input_str.size());
  ofs.close();
}

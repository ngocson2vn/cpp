#include <iostream>
#include <fstream>
#include <string>

int main() {
  for (int i = 0; i < 100; i++) {
    std::string filename("/tmp/pilot_request_");
    filename.append(std::to_string(i));
    std::cout << filename << std::endl;
    std::ofstream os(filename, std::ios::binary);
    std::string bs = "TestData_";
    bs.append(std::to_string(i));
    os.write(bs.c_str(), bs.size());
    os.close();
  }
}

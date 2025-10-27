#include <iostream>
#include <fstream>
#include <iomanip>  // For std::setw, std::setfill

int main() {
  std::ifstream file("/bin/ls", std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open /bin/ls" << std::endl;
    return 1;
  }

  char byte;
  while (file.get(byte)) {  // Read byte-by-byte
    printf("\\%02o", static_cast<unsigned char>(byte));
  }

  if (file.bad()) {
    std::cerr << "Error: Failed to read file" << std::endl;
    return 1;
  }

  std::cout << std::endl;  // Optional: End with newline
  return 0;
}
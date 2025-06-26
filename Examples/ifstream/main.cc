#include <iostream>
#include <string>
#include <memory>
#include <fstream>


int main(int argc, char** argv) {
  std::string file_path = "/usr/bin/ls";
  std::unique_ptr<char> buffer;
  std::ifstream reader(file_path, std::ios::binary);

  // Get the number of bytes
  reader.seekg(0, reader.end);
  int length = reader.tellg();
  reader.seekg(0, reader.beg);

  // Allocate buffer
  buffer.reset(new char[length]);

  // Read data to buffer
  reader.read(buffer.get(), length);
  reader.close();

  // Create a string from buffer
  std::string bs(buffer.get(), length);

  printf("File size: %lu bytes\n", bs.size());
}

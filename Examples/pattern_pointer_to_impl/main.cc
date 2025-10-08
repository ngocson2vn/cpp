#include <iostream>
#include "logger.h"

int main() {
  std::string logFilePath = "./stdout.log";
  Logger logger(logFilePath);
  logger.setLevel(1);
  for (unsigned i = 0; i < 1010; i++) {
    logger.log(1, std::string("Test message ").append(std::to_string(i)));
  }
  std::cout << "DONE" << std::endl;
}

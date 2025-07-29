#include <chrono>
#include <iostream>

int main() {
  // Get current time as a time_point
  auto now = std::chrono::system_clock::now();
  
  // Convert to epoch time in seconds
  auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  
  std::cout << "Epoch time: " << epoch_time << std::endl;
  return 0;
}
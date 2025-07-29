#include <iostream>
#include <filesystem>
#include <fstream>

std::string genTempFile() {
  // Get current time as a time_point
  auto now = std::chrono::system_clock::now();
  
  // Convert to epoch time in seconds
  auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  std::string tempName = std::string("tempfile.").append(std::to_string(epoch_time));
  std::filesystem::path tempPath = std::filesystem::temp_directory_path() / tempName;
  return tempPath.string();
}

int main() {
  std::filesystem::path currentPath = std::filesystem::current_path();
  std::cout << "cwd: " << currentPath.string() << std::endl;

  std::string tempFile = genTempFile();
  std::cout << "tempFile: " << tempFile << std::endl;

  return 0;
}
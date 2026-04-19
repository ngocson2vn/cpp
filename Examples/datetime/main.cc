#include <chrono>
#include <string>
#include <cstdio>
#include <iostream>

std::string timestamp() {
  // 1. Get current time as a time_point
  auto now = std::chrono::system_clock::now();

  // 2. Convert to time_t (legacy timestamp)
  std::time_t t_c = std::chrono::system_clock::to_time_t(now);

  // 3. Convert to broken-down time (local)
  std::tm* now_tm = std::localtime(&t_c);

  // 4. Format using strftime
  char buffer[80];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", now_tm);

  return std::string(buffer);
}

int main() {
  auto ts = timestamp();
  printf("Now: %s\n", ts.c_str());
  return 0;
}
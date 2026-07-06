#include <iostream>
#include "time_cost.h"
#include <chrono>
#include <thread>

int main() {
  perf::TimeCost tc;
  std::this_thread::sleep_for(std::chrono::seconds(3));
  std::cout << tc.get_elapsed() / 1000 << " ms" << std::endl;
}

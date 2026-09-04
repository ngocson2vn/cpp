#include <chrono>

class Timer {
public:
  Timer()
      : t0(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())) {}

  uint64_t elapsed_time() {
    auto t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    return (t1 - t0).count();
  }

private:
  std::chrono::nanoseconds t0;
};

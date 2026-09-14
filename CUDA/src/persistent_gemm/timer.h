#include <chrono>

class Timer {
public:
  Timer()
      : t0(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())) {}

  float elapsed_time_ms() {
    auto t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    return (t1 - t0).count() / 1e6;
  }

  float elapsed_time_us() {
    auto t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    return (t1 - t0).count() / 1e3;
  }

  uint64_t elapsed_time_ns() {
    auto t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    return (t1 - t0).count();
  }

private:
  std::chrono::nanoseconds t0;
};

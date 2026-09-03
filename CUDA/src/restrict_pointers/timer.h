#include <chrono>

class Timer {
public:
  Timer() : tp_(std::chrono::system_clock::now()) {}

  uint64_t elapsed_time() {
    auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    auto start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        tp_.time_since_epoch());
    auto d = end_ns - start_ns;
    return d.count();
  }

private:
  std::chrono::system_clock::time_point tp_;
};

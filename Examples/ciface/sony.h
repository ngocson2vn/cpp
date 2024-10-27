#include <functional>

namespace sony {
  void register_abort_handler(std::function<void(void)> abort_handler);
  void play();
}

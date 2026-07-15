#include <bitset>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

class ConfigurableWorkerManager {
private:
  int num_workers_;
  int concurrency_;

  // Each worker gets its own 64-bit integer for state tracking
  std::vector<uint64_t> worker_states_;

  // A mask representing a state where all allowed slots are full.
  // For example, if concurrency_ = 4, worker_full_mask_ = 0000...1111 (0xF)
  uint64_t worker_full_mask_;

public:
  ConfigurableWorkerManager(int workers, int concurrency)
      : num_workers_(workers), concurrency_(concurrency) {

    if (concurrency <= 0 || concurrency > 64) {
      throw std::invalid_argument(
          "Requests per worker must be between 1 and 64.");
    }

    worker_states_.resize(num_workers_, 0);

    // Generate the bitmask for a "full" worker
    worker_full_mask_ = (~0ULL) >> (64 - concurrency);
  }

  // 1. Check if a worker has at least one available slot
  bool isWorkerAvailable(int worker_id) const {
    if (worker_id < 0 || worker_id >= num_workers_)
      return false;

    // If the worker's state matches the worker_full_mask_, all slots are occupied
    return worker_states_[worker_id] != worker_full_mask_;
  }

  // 2. Allocate the first available slot for a worker
  // Inside your ConfigurableWorkerManager class:
  int allocateSlot(int worker_id) {
    if (!isWorkerAvailable(worker_id)) {
      return -1;
    }

    uint64_t &worker_state = worker_states_[worker_id];

    // Invert the state so 0s (free slots) become 1s
    int slot = __builtin_ctzll(~worker_state);

    // Set the bit to mark the slot as occupied
    worker_state |= (1ULL << slot);

    return slot;
  }

  // 3. Free a specific slot for a worker
  void freeSlot(int worker_id, int slot) {
    if (worker_id < 0 || worker_id >= num_workers_)
      return;
    if (slot < 0 || slot >= concurrency_)
      return;

    // Clear the specific bit to 0 using AND with NOT
    worker_states_[worker_id] &= ~(1ULL << slot);
  }

  // Helper to visualize the state of all workers
  void printState() const {
    std::cout << "--- Current State ---\n";
    for (int i = 0; i < num_workers_; ++i) {
      // Only print the relevant number of bits
      std::cout << "Worker " << i << ": ";
      for (int bit = concurrency_ - 1; bit >= 0; --bit) {
        std::cout << ((worker_states_[i] & (1ULL << bit)) ? '1' : '0');
      }
      std::cout << "\n";
    }
  }
};

int main() {
  // Configure: 3 workers, each handling 6 requests
  ConfigurableWorkerManager manager(3, 6);

  manager.allocateSlot(0);
  manager.allocateSlot(0);
  manager.allocateSlot(2);
  manager.printState();

  std::cout << "\nIs Worker 0 available? "
            << (manager.isWorkerAvailable(0) ? "Yes" : "No") << "\n";

  // Fill up worker 0
  manager.allocateSlot(0);
  manager.allocateSlot(0);
  manager.allocateSlot(0);
  manager.allocateSlot(0);

  std::cout << "Is Worker 0 available after filling? "
            << (manager.isWorkerAvailable(0) ? "Yes" : "No") << "\n";

  manager.printState();

  return 0;
}
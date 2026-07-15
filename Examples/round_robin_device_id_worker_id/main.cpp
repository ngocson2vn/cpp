#include <cstdio>
#include <cstdlib>
#include <utility>

int32_t current_device_id_ = 0;
int32_t current_worker_id_ = 0;

std::pair<int32_t, int32_t> GetDeviceAndWorkerIDs(int32_t device_count_, int32_t worker_count_) {
  int32_t device_id = current_device_id_;
  int32_t worker_id = current_worker_id_;

  current_device_id_ = (current_device_id_ + ((current_worker_id_ + 1) == worker_count_)) % device_count_;
  current_worker_id_ = (current_worker_id_ + 1) % worker_count_;

  return {device_id, worker_id};
}

int main() {
  int32_t device_count_ = 4;
  int32_t worker_count_ = 4;
  printf("device_count_ = %d worker_count_ = %d\n", device_count_, worker_count_);
  for (int i = 0; i < 18; i++) {
    auto [device_id, worker_id] = GetDeviceAndWorkerIDs(device_count_, worker_count_);
    printf("device_id=%d worker_id=%d\n", device_id, worker_id);
  }
}

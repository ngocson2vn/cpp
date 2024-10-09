#include <iostream>
#include <climits>
#include <cstring>
#include <bitset>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <type_traits>
#include "folly/include/folly/portability/Builtins.h"

std::mutex global_mutex;
std::mutex device_mutex;

#define LOG_INFO(...) \
do { \
  std::lock_guard<std::mutex> lk(global_mutex); \
  fprintf(stdout, __VA_ARGS__); \
  fprintf(stdout, "\n"); \
} while(0)

// #define _MSC_VER
// #include <intrin.h>

#define MAX_DEVICE_COUNT 8
#define MAX_DEVICE_SHARE_RATIO 64
#define INVALID_DEVICE_ID -1

using DeviceId = int32_t;

template <size_t bitcap, typename... IntTypes>
struct IntRequiredImpl;

// Recursively select the most appropriate type.
// That is, the number of bits is just less than or equal to the size of the
// type IntRequiredImpl<bitcap, IntType> is a buttom of the selection.
template <size_t bitcap, typename IntType>
struct IntRequiredImpl<bitcap, IntType> {
  using type = IntType;

  // avoid overflow
  static_assert(bitcap <= sizeof(type) * CHAR_BIT, "bitcap is overflow");
};

// Check if the length of SmallInt can cover bitcap.
// Use SmallInt When bitcap <= size of SmallInt.
// Otherwise recursively search next type.
template <size_t bitcap, typename SmallInt, typename... LargeInts>
struct IntRequiredImpl<bitcap, SmallInt, LargeInts...> {
  using type = std::conditional_t<bitcap <= sizeof(SmallInt) * CHAR_BIT, SmallInt, typename IntRequiredImpl<bitcap, LargeInts...>::type>;
};

using BitMap = typename IntRequiredImpl<MAX_DEVICE_COUNT, std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>::type;

const int32_t device_count_ = 1;
const int32_t device_share_ratio_ = 2;
const int32_t kConcurrency = 16;
uint32_t priority_search_id_ = 0;
uint32_t thread_counter = 0;

BitMap device_unexclusive_bitmap_;
BitMap device_has_quota_bitmap_;
BitMap quota_minus_one_to_device_bitmap_[MAX_DEVICE_SHARE_RATIO];
const BitMap all_device_mask_ = (device_count_ == sizeof(BitMap) * CHAR_BIT) ? ~static_cast<BitMap>(0) : (static_cast<BitMap>(1) << device_count_) - 1;


BitMap MakeDeviceToken(DeviceId device_id) {
  return static_cast<BitMap>(1) << device_id;
}

void OccupyDevice(DeviceId device_id, int32_t quota_minus_one,
                  bool exclusive) {
  const auto device_token = MakeDeviceToken(device_id);
  quota_minus_one_to_device_bitmap_[quota_minus_one] &= ~device_token;
  if (quota_minus_one > 0) {
    quota_minus_one_to_device_bitmap_[quota_minus_one - 1] |= device_token;
  } else {
    device_has_quota_bitmap_ &= ~device_token;
  }

  if (exclusive) {
    device_unexclusive_bitmap_ &= ~device_token;
  }
}

DeviceId AcquireAnyDevice(const bool exclusive) {
  int quota_minus_one = static_cast<int>(device_share_ratio_) - 1; // 16 - 1 = 15
  while (quota_minus_one >= 0) {
    auto quota_bits = static_cast<int64_t>(quota_minus_one_to_device_bitmap_[quota_minus_one]);
    // quota_minus_one_to_device_bitmap_[15]

    auto ts = ((quota_bits << (device_count_ - priority_search_id_)) | (quota_bits >> priority_search_id_)) & all_device_mask_;
    DeviceId device_id = static_cast<DeviceId>(-1);
    while (ts > 0) {
      auto delta = __builtin_ffsll(ts);
      LOG_INFO("delta = %d", delta);
      device_id = (device_id + delta + priority_search_id_) % (device_count_);
      if (device_unexclusive_bitmap_ & MakeDeviceToken(device_id)) {
        OccupyDevice(device_id, quota_minus_one, exclusive);
        priority_search_id_ = (priority_search_id_ + 1) % device_count_;
        return static_cast<DeviceId>(device_id);
      }
      ts >>= delta;
    }
    --quota_minus_one;
  }

  return INVALID_DEVICE_ID;
}

void initialize() {
  device_has_quota_bitmap_ = all_device_mask_;
  device_unexclusive_bitmap_ = all_device_mask_;
  std::memset(quota_minus_one_to_device_bitmap_, 0, sizeof(BitMap) * (device_share_ratio_));
  quota_minus_one_to_device_bitmap_[device_share_ratio_ - 1] = all_device_mask_;
}

void line_break() {
  std::lock_guard<std::mutex> lk(global_mutex);
  thread_counter++;
  if (thread_counter == kConcurrency) {
    fprintf(stdout, "\n");
    thread_counter = 0;
  }
}

void do_task(int idx) {
  DeviceId device_id = INVALID_DEVICE_ID;
  {
    std::lock_guard<std::mutex> lk(device_mutex);
    device_id = AcquireAnyDevice(false);
  }

  while (true) {
    LOG_INFO("[tidx=%d] device_id = %d", idx, device_id);
    line_break();
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }
}

int main(int argc, char** argv) {
  initialize();
  int quota_minus_one = device_share_ratio_ - 1;
  std::cout << "device_share_ratio_ = " << device_share_ratio_ << std::endl;
  std::cout << "all_device_mask_ = " << std::bitset<8>(all_device_mask_) << std::endl;
  std::cout << "device_has_quota_bitmap_ = " << std::bitset<8>(device_has_quota_bitmap_) << std::endl;
  std::cout << "quota_minus_one_to_device_bitmap_[" << quota_minus_one << "] = " << std::bitset<8>(quota_minus_one_to_device_bitmap_[quota_minus_one]) << std::endl;

  std::vector<std::thread> threads;
  threads.reserve(kConcurrency);

  for (int i = 0; i < kConcurrency; i++) {
    LOG_INFO("Start thread %d", i);
    threads[i] = std::thread(do_task, i);
  }

  for (int i = 0; i < threads.size(); i++) {
    threads[i].join();
  }

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(60));
  }
  std::cout << "DONE!\n"; 
}

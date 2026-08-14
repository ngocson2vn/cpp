#pragma once

#include <memory>
#include <vector>

namespace gpu {
namespace monitor {

class GpuMonitorImpl;

class GpuMonitor {
public:
  GpuMonitor();

  ~GpuMonitor();

  // start background monitor thread
  void start();

  int get_device_count();

  const char *get_device_uuid(unsigned int device_idx) const;

  std::vector<int> get_device_ids();

private:
  std::unique_ptr<GpuMonitorImpl> impl;
};

} // namespace monitor
} // namespace gpu

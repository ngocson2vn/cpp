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
  void start(int dev_idx);

private:
  std::unique_ptr<GpuMonitorImpl> impl;
};

} // namespace monitor
} // namespace gpu

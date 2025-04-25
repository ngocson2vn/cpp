#include <iostream>
#include <string>
#include <type_traits>

struct DeviceProperties {
  std::string name;
  int computeMajor;
  int computeMinor;
  double peakFp16TensorCoreFlops;
  double peakMemoryBwBytesPerSec;
};

template <typename T>
std::string format(std::string key, T value) {
  return "\"" + key + "\": " + std::to_string(value);
}

std::string format(std::string key, std::string value) {
  return "\"" + key + "\": " + "\"" + value + "\"";
}

std::string createJson(const DeviceProperties& devProp) {
  std::string retJson = "{";
  retJson += format("name", devProp.name);
  retJson += ", " + format("computeMajor", devProp.computeMajor);
  retJson += ", " + format("computeMinor", devProp.computeMinor);
  retJson += ", " + format("peakFp16TensorCoreFlops", devProp.peakFp16TensorCoreFlops);
  retJson += ", " + format("peakMemoryBwBytesPerSec", devProp.peakMemoryBwBytesPerSec);
  retJson += "}";

  return retJson;
}

int main(int argc, char** argv) {
  DeviceProperties dp;
  dp.name = "NVIDIA A10";
  dp.computeMajor = 8;
  dp.computeMinor = 6;
  dp.peakFp16TensorCoreFlops = 125 * 1e12f;
  dp.peakMemoryBwBytesPerSec = 600 * 1e9f;

  // if (std::is_same<std::string, std::remove_reference_t<decltype(dp.name)>>()) {
  //   std::cout << "OK" << std::endl;
  // }

  std::string ret = createJson(dp);
  std::cout << ret << std::endl;
}

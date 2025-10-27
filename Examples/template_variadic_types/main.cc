#include <iostream>
#include <vector>

namespace runtime {

class ModuleMgr {
 public:
  template <typename... Arg>
  bool call(void* funcHandle, Arg... arg) {

    using FuncPtrType = void (*)(Arg...);
    auto funcPtr = reinterpret_cast<FuncPtrType>(funcHandle);
    (funcPtr)(arg...);

    return true;
  }
};

} // namespace runtime

using namespace runtime;

void printVec(float* vecPtr, std::size_t numElements) {
  printf("Vec: ");
  for (std::size_t i = 0; i < numElements; i++) {
    printf("%.00f ", vecPtr[i]);
  }
  printf("\n");
}

int main() {
  std::vector<float> vec{1.0, 2.0, 3.0, 4.0, 5.0};

  ModuleMgr mod;
  void* funcHandle = reinterpret_cast<void*>(printVec);
  mod.call(funcHandle, vec.data(), vec.size());
}

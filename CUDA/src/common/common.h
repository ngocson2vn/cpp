#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>

#include <cuda_runtime.h>

template <typename T>
using HVec = std::vector<T>;

template <typename T>
class DVec {
 public:
  DVec() = default;
  
  DVec(T* ptr, std::size_t size) : ptr_(ptr), size_(size) {}

  DVec(std::size_t size, T initVal);

  DVec(const HVec<T>& hVec);

  ~DVec() {
    if (!ptr_) {
      cudaFree(ptr_);
    }
  }

  T* data() { return ptr_; };
  std::size_t size() { return size_; }
  bool empty() { return size_ == 0; }

 private:
  T* ptr_;
  std::size_t size_ = 0;
};

template <typename T>
struct HVecDVec {
  HVec<T> hVec;
  DVec<T> dVec;
};

template <typename T> 
DVec<T> genDVec(uint32_t numElems, const std::string& name);
#pragma once

#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(callExpr)                                             \
  do {                                                                         \
    auto status = callExpr;                                                    \
    if (status != cudaSuccess) {                                               \
      auto errorName = cudaGetErrorName(status);                               \
      auto errorStr = cudaGetErrorString(status);                              \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, errorName,         \
              errorStr);                                                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)


template <typename T>
class DVec;

template <typename T>
class HVec {
 public:
  HVec() = default;

  HVec(std::size_t numElems, T initVal = T(0))
    : vec(numElems, initVal) {}

  T& operator[](std::size_t idx) const {
    return const_cast<T&>(vec[idx]);
  }

  T* data() const { return const_cast<T*>(vec.data()); }

  std::size_t size() const { return vec.size(); }

  bool empty() const { return vec.empty(); }

  DVec<T> gpu() const { return DVec<T>(*this); }

 private:
  std::vector<T> vec;
};

template <typename T>
class DVec {
 public:
  DVec() = default;

  DVec(const HVec<T>& hVec);

  DVec(std::size_t size, T initVal) : DVec(HVec<T>(size, initVal)) {}

  DVec(T* ptr, std::size_t size) : ptr_(ptr), size_(size) {}

  ~DVec() {
    if (!ptr_) {
      cudaFree(ptr_);
    }
  }

  T* data() { return ptr_; };

  std::size_t size() { return size_; }

  bool empty() { return size_ == 0; }

  HVec<T> cpu();

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
HVec<T> genHVec(uint32_t numElems, const std::string& name);

template <typename T> 
DVec<T> genDVec(uint32_t numElems, const std::string& name);

template <typename T>
bool diff(const HVec<T>& a, const HVec<T>& b);
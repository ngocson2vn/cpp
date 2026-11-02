#include "common.h"

#include <cstdio>
#include <cstdint>

#include <random>
#include <memory>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

template <typename T>
DVec<T>::DVec(const HVec<T>& hVec) {
  if (hVec.empty()) return;

  T* devPtr = nullptr;
  auto numBytes = hVec.size() * sizeof(T);
  auto status = cudaMalloc(&devPtr, numBytes);
  if (status != cudaSuccess) {
    auto errorName = cudaGetErrorName(status);
    auto errorString = cudaGetErrorString(status);
    fprintf(
      stderr,
      "ERROR: Failed to allocate %zu bytes on device %s: %s\n",
      errorName,
      errorString,
      numBytes
    );
    std::exit(1);
  }

  status = cudaMemcpy(devPtr, hVec.data(), numBytes, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    auto errorName = cudaGetErrorName(status);
    auto errorString = cudaGetErrorString(status);
    fprintf(
      stderr,
      "ERROR: Failed to copy data from host to device %s: %s\n",
      errorName,
      errorString,
      numBytes
    );
    std::exit(1);
  }

  ptr_ = devPtr;
  size_ = hVec.size();
}

template <typename T>
HVec<T> DVec<T>::cpu() {
  HVec<T> hVec(size_, T(0));

  auto numBytes = size_ * sizeof(T);
  auto status = cudaMemcpy(hVec.data(), ptr_, numBytes, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    auto errorName = cudaGetErrorName(status);
    auto errorString = cudaGetErrorString(status);
    fprintf(
      stderr,
      "ERROR: Failed to copy data from device to host %s: %s\n",
      errorName,
      errorString,
      numBytes
    );
    std::exit(1);
  }

  return hVec;
}

template HVec<float> DVec<float>::cpu();


template <typename HVec>
void regenHVec(HVec& hVec, const std::string vecFile) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for (uint32_t i = 0; i < hVec.size(); i++) {
    hVec[i] = dist(gen);
  }

  std::ofstream ofs(vecFile);
  if (ofs.good()) {
    std::size_t hVecSize = hVec.size();
    ofs.write(reinterpret_cast<char*>(&hVecSize), sizeof(hVecSize));
    ofs.write(reinterpret_cast<char*>(hVec.data()), hVecSize * sizeof(hVec[0]));
    ofs.close();
    printf("Vector size: %zu\n", hVecSize);
    printf("Saved vector to file %s\n", vecFile.c_str());
  }
}

template <typename T> 
DVec<T> genDVec(uint32_t numElems, const std::string& name) {
  HVec<T> hVec(numElems, T(0));

  std::string vecFile = name + ".dat";
  if (fs::exists(vecFile)) {
    std::ifstream ifs(vecFile);
    if (ifs.good()) {
      std::size_t hVecSize = 0;
      char header[sizeof(hVecSize)];
      if (ifs.read(&header[0], sizeof(hVecSize))) {
        hVecSize = *reinterpret_cast<std::size_t*>(&header[0]);
      } else {
        fprintf(stderr, "ERROR: Failed to read vector size from %s\n", vecFile.c_str());
      }

      if (hVecSize != numElems) {
        regenHVec(hVec, vecFile);
      } else {
        ifs.read(reinterpret_cast<char*>(hVec.data()), hVec.size() * sizeof(hVec[0]));
        if (!!ifs) {
          printf("Loaded vector from file %s\n", vecFile.c_str());
          printf("Vector size: %zu\n", hVecSize);
        } else {
          fprintf(stderr, "ERROR: Failed to read vector elements from %s\n", vecFile.c_str());
        }
      }
    } else {
      fprintf(stderr, "ERROR: Failed to open file %s\n", vecFile.c_str());
    }
  } else {
    regenHVec(hVec, vecFile);
  }

  DVec<T> dVec(hVec);
  return dVec;
}

template DVec<float> genDVec<float>(uint32_t numElems, const std::string& name);
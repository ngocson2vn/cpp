#include <cstdio>
#include <cstdint>
#include <cassert>

#include "constants.h"
#include <common/common.h>
#include <torch_api/torch_api.h>

#include "persistent_gemm.cuh"

int main(int argc, char** argv) {
  // Generate input matrices
  HVec<float> A = genHVec<float>(M * K, "A");
  HVec<float> B = genHVec<float>(K * N, "B");
  HVec<float> gResult = gpuGemm(A, B, M, N, K);
  HVec<float> tResult = torchGemm(A, B, M, N, K);

  // // Test if diff works as expected
  // gResult[10] = gResult[10] + 0.001;
  // bool ok = diff(gResult, tResult);
  // printf("%s\n", (ok ? "PASSED" : "FAILED"));

  printf("\n");
  printf("Matrix C: %dx%d\n", M, N);
  bool ok = diff(gResult, tResult);
  printf("%s\n", (ok ? "PASSED" : "FAILED"));

  return 0;
}

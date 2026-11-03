#include <cstdio>

#include <common/common.h>
#include <torch_api/torch_api.h>

#include "constants.h"
#include "softmax.cuh"

int main(int argc, char** argv) {
  auto hInp = genHVec<float>(M * N, "input");
  auto gResult = gpuSoftmax(hInp, M, N);
  auto tResult = torchSoftmax(hInp, M, N);

  bool ok = diff(gResult, tResult);
  printf("\n%s\n", ok ? "PASSED" : "FAILED");
}
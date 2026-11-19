#include <cstdio>
#include <cstdint>

#include <common/common.h>
#include <torch_api/torch_api.h>

#include "permute.cuh"
#include "constants.h"

int main(int argc, char** argv) {
  auto dataVec = genHVec<float>(M * N * K, "input");
  std::vector<int64_t> dimVec = {M, N, K};
  std::vector<int64_t> permVec = {2, 1, 0};
  auto gResult = gpuPermute(dataVec, dimVec, permVec);
  // const auto ND0 = dimVec[permVec[0]];
  // const auto ND1 = dimVec[permVec[1]];
  // const auto ND2 = dimVec[permVec[2]];
  // printf("ND0 = %d\n", ND0);
  // printf("ND1 = %d\n", ND1);
  // printf("ND2 = %d\n", ND2);
  // printf("\n[SONY] permute output:\n");
  // for (int64_t i = 0; i < ND0; i++) {
  //   for (int64_t j = 0; j < ND1; j++) {
  //     printf("[%f", gResult[i * ND1 * ND2 + j * ND2]);
  //     for (int64_t k = 1; k < ND2; k++) {
  //       printf(", %f", gResult[i * ND1 * ND2 + j * ND2 + k]);
  //     }
  //     printf("]\n");
  //   }
  //   printf("\n");
  // }

  auto tResult = torchPermute(dataVec, dimVec, permVec);

  bool ok = diff(gResult, tResult);
  printf("\n%s\n", ok ? "PASSED" : "FAILED");
}

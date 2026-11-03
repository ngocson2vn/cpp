#include <common/common.h>
#include <torch_api/torch_api.h>

#include "constants.h"

int main(int argc, char** argv) {
  // Generate input matrices
  auto A = genHVec<float>(M * K, "A");
  auto B = genHVec<float>(K * N, "B");
  auto C = torchGemm(A, B, M, N, K);

  printf("\n[TORCH] matrix C %dx%d:\n", M, N);
  for (uint32_t i = 0; i < M; i++) {
    printf("[%f", C[i * N + 0]);
    for (uint32_t j = 1; j < N; j++) {
      printf(", %f", C[i * N + j]);
    }
    printf("]\n");
  }

  return 0;
}

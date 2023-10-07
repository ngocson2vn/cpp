#include "cuda.h"

// Kernel - Adding two matrices MatA and MatB
__global__ void MatAdd(int** MatA, int** MatB, int** MatC) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < 2 && j < 2) MatC[i][j] = MatA[i][j] + MatB[i][j];
}

void mat_add(int** MatA, int** MatB, int** MatC) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (2 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  MatAdd<<<numBlocks, threadsPerBlock>>>(MatA, MatB, MatC);
}
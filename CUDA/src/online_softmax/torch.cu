#include <iostream>
#include <common/common.h>

#include <torch/torch.h>

#include "constants.h"

int main(int argc, char** argv) {
  if (argc > 2) {
    try {
      ROWS = std::stoi(argv[1]);
      COLS = std::stoi(argv[2]);
    } catch(const std::exception& ex) {
      printf("Default ROWS = %d COLS = %d\n", ROWS, COLS);
    }
  }
  assert(COLS >= 32 && "Row size must be >= 32!");
  printf("ROWS = %d COLS = %d\n", ROWS, COLS);

  // Gen input
  DVec<DataType> input = genDVec<DataType>(ROWS * COLS, "input");

  // Allocate output
  DVec<DataType> devOut(ROWS * COLS, DataType(0));

  torch::Device device(c10::kCUDA, 0);
  torch::TensorOptions options(device);
  auto inputTensor = torch::from_blob(
    input.data(),
    {ROWS, COLS},
    {COLS, 1},
    options
  );
  // std::cout << "inputTensor: " << inputTensor << "\n\n";

  auto outTensor = inputTensor.softmax(-1).cpu();
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  auto outData = outTensor.data_ptr<DataType>();

  printf("\n[TORCH] softmax output:\n");
  for (uint32_t i = 0; i < ROWS; i++) {
    printf("[%f", outData[i * COLS + 0]);
    for (uint32_t j = 1; j < COLS; j++) {
      printf(", %f", outData[i * COLS + j]);
    }
    printf("]\n\n");
  }

  return 0;
}

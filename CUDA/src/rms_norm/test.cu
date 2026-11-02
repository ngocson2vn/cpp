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
  printf("ROWS = %d COLS = %d\n", ROWS, COLS);

  DVec<DataType> input = genDVec<DataType>(ROWS * COLS, "input");
  DVec<DataType> gamma = genDVec<DataType>(COLS, "gamma");

  torch::Device device(c10::kCUDA, 0);
  torch::TensorOptions options(device);
  auto inputTensor = torch::from_blob(
    input.data(),
    {ROWS, COLS},
    {COLS, 1},
    options
  );
  // std::cout << "inputTensor: " << inputTensor << "\n\n";

  auto gammaTensor = torch::from_blob(
    gamma.data(),
    {COLS},
    {1},
    options
  );
  // std::cout << "gammaTensor: " << gammaTensor << "\n\n";

  auto outTensor = torch::rms_norm(
    inputTensor,
    {COLS},
    gammaTensor,
    0.0L
  ).cpu();
  // std::cout << "outTensor: " << outTensor << "\n\n";

  auto outData = outTensor.data_ptr<DataType>();

  printf("\n[TORCH] RMSNorm output:\n");
  for (uint32_t i = 0; i < ROWS; i++) {
    printf("[%f", outData[i * COLS + 0]);
    for (uint32_t j = 1; j < COLS; j++) {
      printf(", %f", outData[i * COLS + j]);
    }
    printf("]\n\n");
  }

  return 0;
}

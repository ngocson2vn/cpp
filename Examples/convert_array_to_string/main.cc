#include <iostream>
#include <vector>
#include <string>

using DType = int32_t;

int main(int argc, char** argv) {
  int N = 20;
  std::vector<DType> array;
  for (int i = 1; i < (N + 1); i++) {
    array.push_back(i);
  }

  for (int i = 0; i < N; i++) {
    printf("%ld ", array[i]);
  }
  printf("\n\n");

  std::string raw_bytes(reinterpret_cast<const char*>(array.data()), N * sizeof(DType));
  for (int i = 0; i < raw_bytes.size(); i++) {
    printf("0x%02x ", raw_bytes[i]);
    if (i > 0 && (i + 1) % sizeof(DType) == 0) {
      printf("\n");
    }
  }
  printf("\n\n");

  const DType* a = reinterpret_cast<const DType*>(raw_bytes.data());
  for (int i = 0; i < N; i++) {
    printf("%ld ", a[i]);
  }
  printf("\n");
}

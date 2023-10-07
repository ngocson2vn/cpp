#include <cstdio>
#include <cstdlib>

#include "cuda.h"

void print_mat(int** a, const char* name) {
  printf("%s:\n", name);
  for (size_t i = 0; i < 2; i++) {
    printf("i = %d:", i);
    for (size_t j = 0; j < 2; j++) {
      printf(" %-10d", a[i][j]);
    }
    printf("\n");
  }
}

int main() {
  int** MatA = (int**)malloc(2 * sizeof(int*));
  for (size_t i = 0; i < 2; i++) {
    MatA[i] = (int*)malloc(2 * sizeof(int));
    for (size_t j = 0; j < 2; j++) {
      MatA[i][j] = i + j + 1;
    }
  }

  int** MatB = (int**)malloc(2 * sizeof(int*));
  for (size_t i = 0; i < 2; i++) {
    MatB[i] = (int*)malloc(2 * sizeof(int));
    for (size_t j = 0; j < 2; j++) {
      MatB[i][j] = i + j + 1;
    }
  }

  int** MatC = (int**)malloc(2 * sizeof(int*));
  for (size_t i = 0; i < 2; i++) {
    MatC[i] = (int*)malloc(2 * sizeof(int));
    for (size_t j = 0; j < 2; j++) {
      MatC[i][j] = 0;
    }
  }

  print_mat(MatA, "MatA");
  printf("\n");
  print_mat(MatB, "MatB");
  printf("\n");
  print_mat(MatC, "MatC");
  printf("\n");

  // Matrix addition kernel launch from host code
  mat_add(MatA, MatB, MatC);

  printf("===========================\n");
  print_mat(MatC, "MatC");
  printf("\n");
}
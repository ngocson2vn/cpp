#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void mat_add(int** MatA, int** MatB, int** MatC);

#ifdef __cplusplus
}
#endif
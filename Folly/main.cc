#include <iostream>
#include "folly/container/detail/F14IntrinsicsAvailability.h"

int main(int argc, char** argv) {
  std::cout << "FOLLY_SSE = " << FOLLY_SSE << std::endl;
  std::cout << "FOLLY_SSE_MINOR = " << FOLLY_SSE_MINOR << std::endl;
  std::cout << "FOLLY_F14_VECTOR_INTRINSICS_AVAILABLE = " << FOLLY_F14_VECTOR_INTRINSICS_AVAILABLE << std::endl;
  std::cout << "FOLLY_F14_CRC_INTRINSIC_AVAILABLE = " << FOLLY_F14_CRC_INTRINSIC_AVAILABLE << std::endl;
}

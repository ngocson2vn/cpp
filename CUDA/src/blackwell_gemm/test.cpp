#include <iostream>

int main(int argc, char** argv) {
  // TEST: TMEM start lane and start column per warp
  for (uint32_t TILE_M = 64; TILE_M < 129; TILE_M += 64) {
    std::cout << "TILE_M = " << TILE_M << std::endl;
    const uint32_t ROW_BIT_OFFSET = (TILE_M == 64) ? 21 : 20;
    const uint32_t ROW_MASK = (TILE_M == 64) ? 6291456 : 7340032;
    const uint32_t COL_MASK = (TILE_M == 64) ? 32 : 0;
    for (uint32_t warpIdx = 0; warpIdx < 8; warpIdx++) {
      uint32_t rowOffset = (warpIdx << ROW_BIT_OFFSET) & ROW_MASK;
      uint32_t colOffset = (warpIdx * 8) & COL_MASK;
      std::cout << "warpIdx " << warpIdx << ": Lane " << (rowOffset >> 16) << " Column " << colOffset << std::endl;
    }

    std::cout << std::endl;
  }

  return 0;
}
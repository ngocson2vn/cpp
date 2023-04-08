#include <cstddef>

static const size_t kBlockTrailerSize = 5;

int main() {
    char trailer[kBlockTrailerSize];
    trailer[0] = 0x0;
}
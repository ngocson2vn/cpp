#include <iostream>
#include <limits>

size_t get_bs() {
    int64_t bs = -1;
    std::cout << "bs1 = " << bs << std::endl;
    return bs;
}

int main()
{
    size_t maxvalue = std::numeric_limits<size_t>::max();
    std::cout << "maxvalue = " << maxvalue << std::endl;
    size_t bs = get_bs();
    std::cout << "bs2 = " << bs << std::endl;
    return 0;
}
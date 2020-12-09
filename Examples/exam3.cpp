#include <stdint.h>
#include <iostream>

char* EncodeVarint32(char* dst, uint32_t v)
{
    // Operate on characters as unsigneds
    unsigned char* ptr = reinterpret_cast<unsigned char*>(dst);
    static const int B = 128;
    if (v < (1 << 7))
    {
        *(ptr++) = v;
    }
    else if (v < (1 << 14))
    {
        *(ptr++) = v | B;
        *(ptr++) = v >> 7;
    }
    else if (v < (1 << 21))
    {
        *(ptr++) = v | B;
        *(ptr++) = (v >> 7) | B;
        *(ptr++) = v >> 14;
    }
    else if (v < (1 << 28))
    {
        *(ptr++) = v | B;
        *(ptr++) = (v >> 7) | B;
        *(ptr++) = (v >> 14) | B;
        *(ptr++) = v >> 21;
    }
    else
    {
        *(ptr++) = v | B;
        *(ptr++) = (v >> 7) | B;
        *(ptr++) = (v >> 14) | B;
        *(ptr++) = (v >> 21) | B;
        *(ptr++) = v >> 28;
    }
    return reinterpret_cast<char*>(ptr);
}

int main() {
    char buf[5];
    char* p1 = buf;
    char* p2 = nullptr;
    std::cout << (void*)p1 << std::endl;
    p2 = p1++;
    std::cout << (void*)p2 << std::endl;
    std::cout << (void*)p1 << std::endl;
    EncodeVarint32(buf, 99);
    std::cout << buf << std::endl;
}
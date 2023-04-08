#include <iostream>

int global = 100;

int& setGlobal()
{
    return global;    
}

int main() {
    std::cout << "BEFORE global = " << global << std::endl;
    setGlobal() = 400; // OK
    std::cout << "AFTER global = " << global << std::endl;
}

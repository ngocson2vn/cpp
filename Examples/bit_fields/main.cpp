#include <iostream>
#include <cstdint>


// Layout
/*
┌───────┬───────┬───────┬───────┐───────┬───────┬───────┐───────┐
│///////////////////////////////│       │       │       │///////│
│// Final Padding (:4) /////////│   base_offset_ (:3)   │//Pad//│
│///////////////////////////////│       │       │       │/ :1 //│
└───────┴───────┴───────┴───────┘───────┴───────┴───────┘───────┘
│<----------- 4 bits ---------->│<------- 3 bits ------>│<-1 b->│
*/
struct ControlRegister {
  // 1 bit padding (Bit 0)
  // 3 bits data   (Bits 1-3)
  // 4 bits padding (Bits 4-7)
  uint8_t : 1, base_offset_ : 3, : 4;
  
  // Helper to print the value
  void printValue() {
    std::cout << "Offset: " << (int)base_offset_ << std::endl;
  }
};

int main() {
  ControlRegister reg;
  
  // We can assign 0-7
  reg.base_offset_ = 5; // Binary 101
  reg.printValue();
  
  // CAUTION: Overflow
  reg.base_offset_ = 9; // Binary 1001. The top bit is chopped off.
  // Result is 001 (binary) -> 1
  reg.printValue();

  return 0;
}

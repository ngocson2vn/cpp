#include <iostream>

/*
The Concept: Bit Fields

The syntax : 16 after the variable name is what turns these standard integers into bit fields.
Normally, an int32_t occupies 32 bits (4 bytes) of memory.
However, by adding : 16, you are telling the compiler: "Even though the type is 32-bit, only use 16 bits for this specific variable."

Memory Layout
Because device_id takes 16 bits and context_id takes 16 bits, they sum up to exactly 32 bits.
This means the compiler can pack both members into the space of a single 32-bit integer (4 bytes), 
rather than using 8 bytes (which would happen if you didn't use bit fields).
*/
typedef struct {
  int32_t device_id : 16;
  int32_t context_id : 16;
} DeviceContextId;

/*
In C++, a union is a user-defined datatype in which we can define members of different types of data types just like structures. 
But one thing that makes it different from structures is that the member variables in a union share the same memory location, 
unlike a structure that allocates memory separately for each member variable.
The size of the union is equal to the size of the largest data type.
Memory space can be used by one member variable at one point in time, which means if we assign value to one member variable, 
it will automatically deallocate the other member variable stored in the memory which will lead to loss of data.
*/
using DeviceContextId_U = union {
  DeviceContextId combined_id;
  int32_t id;
};

int main(int argc, char** argv) {
  DeviceContextId_U v;
  v.combined_id = {0, 1};
  std::cout << "&v.combined_id=" << &v.combined_id << std::endl;
  std::cout << "&v.id=" << &v.id << std::endl;
  std::cout << "v.id=" << v.id << std::endl; // 2**16

  std::cout << std::endl;

  DeviceContextId_U v2;
  v2.combined_id = {0, 2};
  std::cout << "&v2.combined_id=" << &v2.combined_id << std::endl;
  std::cout << "&v2.id=" << &v2.id << std::endl;
  std::cout << "v2.id=" << v2.id << std::endl; // 2**17
}

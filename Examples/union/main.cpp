#include <iostream>

typedef struct {
  int32_t device_id : 16;
  int32_t context_id : 16;
} DeviceContextId;

using DeviceContextId_U = union {
  DeviceContextId combined_id;
  int32_t id;
};

int main(int argc, char** argv) {
  DeviceContextId_U v;
  v.combined_id = {0, 1};
  std::cout << "&v.combined_id=" << &v.combined_id << std::endl;
  std::cout << "&v.id=" << &v.id << std::endl;
  std::cout << "v.id=" << v.id << std::endl;

  DeviceContextId_U v2;
  v2.combined_id = {0, 2};
  std::cout << "&v2.combined_id=" << &v2.combined_id << std::endl;
  std::cout << "&v2.id=" << &v2.id << std::endl;
  std::cout << "v2.id=" << v2.id << std::endl;
}

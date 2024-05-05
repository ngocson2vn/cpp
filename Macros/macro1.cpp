#include <iostream>

#define REGISTER_LOCAL_DEVICE_FACTORY(device_factory, ...) INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY(device_factory, __COUNTER__, ##__VA_ARGS__)

#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY(device_factory, ctr, ...) \
  static device_factory                                                  \
  INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr)(__VA_ARGS__)

#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr) ___##ctr##__object_

class Dummy {
  public:
    Dummy() {
      std::cout << "Default Dummy Constructor" << std::endl;
    }

    Dummy(int priority) : priority_(priority) {
      std::cout << "Dummy(int priority) Constructor" << std::endl;
      std::cout << "priority_ = " << priority_ << std::endl;
    }
  
  private:
    int priority_;
};

REGISTER_LOCAL_DEVICE_FACTORY(Dummy, 210);

int main() {
  std::cout << "== main() ==\n";
  return 0;
}
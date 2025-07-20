#include <iostream>
#include <string>

template <typename T, typename Tout = T>
struct FAParams {
  T* q_ptr;
  Tout* o_ptr;
};

template <typename T>
class FAKernel {
 public:
  using Tout = typename std::conditional<std::is_same<T, uint8_t>::value, float, T>::type;
  using Params = FAParams<T, Tout>;
};

template <typename T>
void display_type() {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  // std::cout << "tmp = " << tmp << std::endl;
  std::string type = tmp.substr(4, tmp.size() - 5);
  std::cout << "T = " << type << std::endl;
}

template <typename T>
void display_inferred_type() {
  using Tout = typename std::conditional<std::is_same<T, uint8_t>::value, float, T>::type;
  display_type<Tout>();
}

void dummy_func(const char* str) {
  std::cout << str << std::endl;
}

int main(int argc, char** argv) {
  std::string test_str("test");
  display_type<decltype(test_str)>();
  display_type<decltype(dummy_func)>();

  // bool ok = std::is_same<unsigned char, uint8_t>::value;
  // std::cout << "ok = " << ok << std::endl;

  display_inferred_type<uint8_t>();
  using Kernel = FAKernel<uint8_t>;
  display_type<Kernel::Params>();
}

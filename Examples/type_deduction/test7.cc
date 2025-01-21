// remove_reference
// Ref: https://en.cppreference.com/w/cpp/types/remove_reference

#include <type_traits>

template <typename T>
struct TD;

struct Sony {
  int data;
};

int main(int argc, char** argv) {
  Sony s{100};
  
  Sony& lvalue_ref_s = s;
  // TD<decltype(lvalue_ref_s)> td;

  Sony&& rvalue_ref_s = Sony{200};
  // TD<decltype(rvalue_ref_s)> td;

  using PlainType1 = std::remove_reference<decltype(s)>::type;
  TD<PlainType1> td1;

  using PlainType2 = std::remove_reference<decltype(lvalue_ref_s)>::type;
  TD<PlainType2> td2;

  using PlainType3 = std::remove_reference<decltype(rvalue_ref_s)>::type;
  TD<PlainType3> td3;
}

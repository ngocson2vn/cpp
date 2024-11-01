#include "sony.h"
#include <cmath>
#include <iostream>

namespace sony {
namespace detail {

template <typename T>
struct TypeDetector;

class Foo {
 public:
  void compute() {
    sony::DT sony_v{10};
    // TypeDetector<decltype(sony_v)> td;
    std::cout << exp(sony_v.value) << std::endl;
  }
};

}
}

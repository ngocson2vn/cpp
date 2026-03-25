#include <iostream>
#include <vector>

#include "hashing.h"

// main
//   -> sony::hash_fn(vec)
//     -> do_hash(vec)
// 
// Since the argument `vec` is a std::vector<int>, ADL will search for a `do_hash` function overload in 
// the same namespace `std`. Therefore, the following namespace must be `std` to make ADL work.
namespace std {
  template <typename T> 
  std::size_t do_hash(const std::vector<T>& vec) {
    auto hash_fn = std::hash<T>();
    std::size_t ret = 0;
    for (const auto& e : vec) {
      ret += hash_fn(e);
    }

    return ret;
  }
}


int main(int argc, char** argv) {
  std::vector<int> vec = {0, 1, 2, 3};
  auto h = sony::hash_fn(vec);
  std::cout << "hash = " << h << std::endl;
}


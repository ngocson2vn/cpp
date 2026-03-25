#include <functional>

namespace sony {

template <typename T> 
std::size_t do_hash(const T& value) {
  return std::hash<T>()(value);
}

template <typename T> 
std::size_t hash_fn(T value) {
  return do_hash(value);
}

}

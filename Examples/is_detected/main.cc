#include <iostream>
#include <string>
#include <unordered_map>
#include <functional>

// examples/bazel-examples/external/llvm-project/llvm/include/llvm/ADT/STLExtras.h
namespace detail {

template <class, template <class...> class Op, class... Args> 
struct detector {
  using value_t = std::false_type;
};

template <template <class...> class Op, class... Args>
struct detector<std::void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
};

} // end namespace detail

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<void, Op, Args...>::value_t;

class TypeID {
 public:
  TypeID() = default;
  TypeID(std::size_t id, std::string name) : id_(id), name_(name) {

  }

  static std::size_t hash(const std::string& name) {
    static std::hash<std::string> hash_fn = std::hash<std::string>();
    return hash_fn(name);
  }

  template<typename T>
  static TypeID get() {
    static std::unordered_map<std::size_t, std::string> type_map;
    std::string func_name = __PRETTY_FUNCTION__;
    std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
    std::string type_name = tmp.substr(4, tmp.size() - 5);
    std::cout << "T = " << type_name << std::endl;
    std::size_t id = TypeID::hash(type_name);
    if (auto it = type_map.find(id); it != type_map.end()) {
      return TypeID(it->first, it->second);
    } else {
      type_map.insert({id, type_name});
    }

    return TypeID(id, type_name);
  }

  friend std::ostream& operator<<(std::ostream& os, const TypeID& rhs) {
    os << "id = " << rhs.id_ << ", name = " << rhs.name_;
    return os;
  }

 private:
  std::size_t id_;
  std::string name_;
};

class LmhloOp {
 public:
  static TypeID getInterfaceID() {
    return TypeID::get<LmhloOp>();
  }
};

template <typename T, typename... Args>
using has_get_interface_id = decltype(T::getInterfaceID());

template <typename T>
void display_type() {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  // std::cout << "tmp = " << tmp << std::endl;
  std::string type = tmp.substr(4, tmp.size() - 5);
  std::cout << "T = " << type << std::endl;
}

int main(int argc, char** argv) {
  // Test if has_get_interface_id works
  // display_type<has_get_interface_id<LmhloOp>>();

  TypeID tid = LmhloOp::getInterfaceID();
  std::cout << tid << std::endl;
  if constexpr(is_detected<has_get_interface_id, LmhloOp>::value) {
    std::cout << "OK" << std::endl;
  } else {
    std::cout << "NG" << std::endl;
  }
}

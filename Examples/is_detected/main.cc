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

template <typename T, typename... Args>
using has_get_interface_id = decltype(T::getInterfaceID());

// examples/bazel-examples/external/llvm-project/mlir/include/mlir/Support/InterfaceSupport.h
template <typename T, typename... Args>
using has_get_interface_id = decltype(T::getInterfaceID());

template <typename T>
using detect_get_interface_id = is_detected<has_get_interface_id, T>;

/// Template utility that computes the number of elements within `T` that
/// satisfy the given predicate.
// The primary template handles the base case: when the parameter pack Ts... is empty
template <template <class> class Pred, size_t N, typename... Ts>
struct count_if_t_impl {
  static constexpr auto value = std::integral_constant<size_t, N>::value;
};

/*
This is a partial specialization of the primary template.
It is more specific because it constrains the parameter pack Ts... from the primary template to a non-empty pack, where:
- The first type in the pack is explicitly named T.
- The remaining types are captured as Us... (another variadic pack).

This specialization matches when there is at least one type in the parameter pack (i.e., T, Us...).
*/
template <template <class> class Pred, size_t N, typename T, typename... Us>
struct count_if_t_impl<Pred, N, T, Us...> {
  static constexpr auto value = std::integral_constant<size_t, count_if_t_impl<Pred, N + (Pred<T>::value ? 1 : 0), Us...>::value>::value;
};

template <template <class> class Pred, typename... Ts>
using count_if_t = count_if_t_impl<Pred, 0, Ts...>;
// count_if_t_impl<detect_get_interface_id, 0, <AddOp, MulOp>>;

template <typename... Types>
constexpr auto num_interface_types_v = count_if_t<detect_get_interface_id, Types...>::value; // Refer to ./detect_get_interface_id.h

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

class AddOp {
 public:
  static TypeID getInterfaceID() {
    return TypeID::get<AddOp>();
  }
};

class MulOp {
 public:
  static TypeID getInterfaceID() {
    return TypeID::get<MulOp>();
  }
};

template <typename T>
void display_type() {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  // std::cout << "tmp = " << tmp << std::endl;
  std::string type = tmp.substr(4, tmp.size() - 5);
  std::cout << "T = " << type << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "======================================================" << std::endl;
  std::cout << "Test is_detected" << std::endl;
  std::cout << "======================================================" << std::endl;
  // Test if has_get_interface_id works
  // display_type<has_get_interface_id<LmhloOp>>();

  TypeID tid = AddOp::getInterfaceID();
  std::cout << tid << std::endl;
  if constexpr(is_detected<has_get_interface_id, AddOp>::value) {
    std::cout << "OK" << std::endl;
  } else {
    std::cout << "NG" << std::endl;
  }
  std::cout << std::endl;

  std::cout << "======================================================" << std::endl;
  std::cout << "Test num_interface_types_v" << std::endl;
  std::cout << "======================================================" << std::endl;
  // display_type<num_interface_types_t<AddOp, MulOp>>();
  constexpr size_t numInterfaces = num_interface_types_v<AddOp, std::string, MulOp, int>;
  std::cout << "numInterfaces = " << numInterfaces << std::endl;
}

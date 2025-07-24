#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>

#if CASE_1
//======================================================================
// Case 1
//======================================================================
template<typename T1, typename T2, typename T3>
struct detector {
  using value_t = std::false_type;
};

template<typename T>
struct detector<typename std::enable_if<T::status>::type, int, T> {
  using value_t = std::true_type;
};

struct Failure {
  static constexpr bool status = false;
};

struct Success {
  static constexpr bool status = true;
};

int main() {
  using AType = detector<void, int, Success>::value_t;
  constexpr bool status1 = AType::value;
  std::cout << "status1 = " << (status1 ? "true" : "false") << std::endl;

  using BType = detector<void, int, Failure>::value_t;
  constexpr bool status2 = BType::value;
  std::cout << "status2 = " << (status2 ? "true" : "false") << std::endl;
}

#elif CASE_2
//======================================================================
// Case 2
//======================================================================
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

template <class, template <class...> class Op, class... Args> 
struct detector {
  using value_t = std::false_type;
};

template <template <class...> class Op, class... Args>
struct detector<std::void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
};

template <typename T, typename... Args>
using has_get_interface_id = decltype(T::getInterfaceID());

class AddOp {
 public:
  static TypeID getInterfaceID() {
    return TypeID::get<AddOp>();
  }
};

int main() {
  using BType = detector<void, has_get_interface_id, AddOp>::value_t;
  constexpr bool status = BType::value;
  std::cout << "status = " << (status ? "true" : "false") << std::endl;
}
#endif
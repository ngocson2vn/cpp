#include <iostream>
#include <string>
#include <vector>


template <typename Container, typename T>
class ClassA {
 public:
  void Compute(Container c, T* out);
};

using VecInt = std::vector<int>;

template <typename T>
class ClassA<VecInt, T> {
 public:
  explicit ClassA(const std::string& name) : name_(name) {}

  void Compute(VecInt data, T* out) {
    T sum = 0;
    for (auto& e : data) {
      sum += e;
    }

    std::cout << "Sum: " << sum << std::endl;
    *out = sum;
  };

 private:
  std::string name_;
};

int main(int argc, char** argv) {
  ClassA<VecInt, int> a("vec_int");
  VecInt data = {0, 1, 3};
  int result = 0;
  a.Compute(data, &result);
}
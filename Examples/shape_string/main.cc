#include <iostream>
#include <string>
#include <vector>

auto shape2str = [](const std::vector<int>& dim_sizes) -> std::string {
  if (dim_sizes.size() == 0) return "()";

  std::string ret_str = std::string("(") + std::to_string(dim_sizes[0]);
  for (int i = 1; i < dim_sizes.size(); i++) {
    ret_str += ", " + std::to_string(dim_sizes[i]);
  }
  ret_str += ")";

  return ret_str;
};

int main(int argc, char** argv) {
  std::vector<int> s1 = {16};
  std::cout << shape2str(s1) << std::endl;
  std::vector<int> s2 = {16, 32};
  std::cout << shape2str(s2) << std::endl;
}

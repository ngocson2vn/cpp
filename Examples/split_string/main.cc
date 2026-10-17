#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
  std::vector<std::string> ret_vec;
  std::size_t slen = str.size();
  std::size_t dlen = delimiter.size();
  std::size_t spos = 0;
  auto cpos = str.find(delimiter, spos);
  while (cpos != std::string::npos) {
    auto len = cpos - spos;
    if (len > 0) {
      ret_vec.push_back(str.substr(spos, len));
    }

    spos = cpos + dlen;
    cpos = str.find(delimiter, spos);
  }

  auto len = slen - spos;
  if (len > 0) {
    ret_vec.push_back(str.substr(spos, len));
  }

  return ret_vec;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  std::cout << "[";

  if (vec.size() > 0) {
    os << vec[0];
  } else {
    os << "]";
    return os;
  }

  for (int i = 1; i < vec.size(); i++) {
    os << ", " << vec[i];
  }
  os << "]";

  return os;
}

int main(int argc, char** argv) {
  const std::string str0 = "d0";
  auto vec0 = split(str0, "x");
  std::cout << "vec0: " << vec0 << std::endl;

  const std::string str1 = "xd1";
  auto vec1 = split(str1, "x");
  std::cout << "vec1: " << vec1 << std::endl;

  const std::string str2 = "xd1x";
  auto vec2 = split(str2, "x");
  std::cout << "vec2: " << vec2 << std::endl;

  const std::string str3 = "xd1xf32";
  auto vec3 = split(str3, "x");
  std::cout << "vec3: " << vec3 << std::endl;

  const std::string str4 = "d0xd1xf32";
  auto vec4 = split(str4, "x");
  std::cout << "vec4: " << vec4 << std::endl;

  return 0;
}
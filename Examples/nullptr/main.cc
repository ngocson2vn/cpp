#include <iostream>
#include <cstring>

void getErrMsg(const char** errMsg) {
  // First time, its value is 0
  const long int v1 = reinterpret_cast<const long int>(*errMsg);
  std::cout << "BEFORE *errMsg = " << v1 << std::endl;

  // Failed to initialize CUDA
  int len = 26;
  *errMsg = new char[len];

  const long int v2 = reinterpret_cast<const long int>(*errMsg);
  std::cout << "AFTER *errMsg = " << v2 << std::endl;

  char* tmpErrMsg = const_cast<char*>(*errMsg);
  const long int v3 = reinterpret_cast<const long int>(tmpErrMsg);
  std::cout << "tmpErrMsg = " << v3 << std::endl;

  std::strcpy(tmpErrMsg, "Failed to initialize CUDA");
}

int main() {
  const char* errMsg = nullptr;
  getErrMsg(&errMsg);
  std::cout << "errMsg: " << errMsg << std::endl;
  return 0;
}
#include <iostream>
#include <string>

template <typename T>
class TypeD;

int main() {
  char data[] = "This is sample data";
  size_t n = sizeof(data);
  char* pdata = data;
  char* limit = pdata + n;
  printf("%p\n", pdata);
  printf("%p\n", limit);
  bool flag = pdata < limit;
  printf("%s\n", flag ? "true" : "false");
  // TypeD<decltype(limit)> t;
}

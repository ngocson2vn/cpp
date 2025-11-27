#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cxxabi.h>  // Required for __cxa_demangle
#include <memory>
#include <string>
#include <vector>

// Helper function to demangle a C++ name
std::string demangle(const char* mangled_name) {
  int status = 0;
  char* res = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
  return (status == 0) ? res : "";
}

std::string get_mangled_name(const char* str) {
  std::string mangled_name;
  std::string tmp_str(str);
  bool found = false;
  for (int i = 0; i < tmp_str.size(); i++) {
    if (!found) {
      if (tmp_str[i] != '(') {
        continue;
      } else {
        found = true;
      }
    }

    if (tmp_str[i] == '+') {
      break;
    }

    if (tmp_str[i] != '(') {
      mangled_name.append(1, tmp_str[i]);
    }
  }

  if (!mangled_name.empty()) {
    return mangled_name;
  }

  return "";
}

void printCallers() {
  void* array[10];
  char** strings;
  int size;

  size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);

  if (strings != NULL) {
    printf("Obtained %d stack frames.\n", size);
    for (int i = 0; i < size; i++) {
      auto mangled_name = get_mangled_name(strings[i]);
      if (!mangled_name.empty()) {
        auto name = demangle(mangled_name.c_str());
        if (!name.empty()) {
          printf("%s\n", name.c_str());
        } else {
          printf("%s\n", strings[i]);
        }
      } else {
        printf("%s\n", strings[i]);
      }
    }
  }

  free(strings);
}

void callee() {
  printCallers();
}

void caller() {
  callee();
}

int main(int argc, char** argv) {
  caller();
  return 0;
}

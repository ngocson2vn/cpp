#include "sony.h"
#include <string>
#include <dlfcn.h>
#include <iostream>

static constexpr char kTargetFuncName[] = "_sony_tf_set_error_message";

typedef void (*ciface)(const char* error_message);

namespace {
  void publish_error(const std::string error) {
    void* func_ptr = dlsym(RTLD_DEFAULT, kTargetFuncName);
    if (func_ptr) {
      std::cout << "Found ciface " << func_ptr << std::endl;
      reinterpret_cast<ciface>(func_ptr)(error.c_str());
    }
  }
}

namespace sony {
  void play() {
    std::string error = "Internal error!";
    publish_error(error);
  }
}
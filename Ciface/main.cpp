#include <cstdio>
#include <string>
#include <dlfcn.h>

#include "sony.h"

typedef const char* (*TfCiface)();

static constexpr char kGetLastTfErrorFuncName[] = "_tf_get_last_error_message";

void LogLastTfError() {
  void* func_ptr = dlsym(RTLD_DEFAULT, kGetLastTfErrorFuncName);
  if (func_ptr) {
    fprintf(stderr, "Found %s at %p\n", kGetLastTfErrorFuncName, func_ptr);
    const char* last_tf_error = reinterpret_cast<TfCiface>(func_ptr)();
    if (last_tf_error) {
      fprintf(stderr, "ERROR: %s\n", last_tf_error);
    }
  }
}

class TfWatcher {
 public:
  TfWatcher() {
    sony::register_abort_handler([](){
      LogLastTfError();
    });
  }
};

static TfWatcher tf_watcher;

int main(int argc, char** argv) {
  sony::play();
}

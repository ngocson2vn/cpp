#include "sony.h"
#include <string>
#include <iostream>

#define EXPORT_CIFACE __attribute__((visibility("default")))

static std::function<void(void)> custom_abort_handle;

static std::string _last_tf_error;

extern "C" {
  EXPORT_CIFACE const char* _tf_get_last_error_message() {
    return _last_tf_error.c_str();
  }
}

namespace {
  void store_error(const std::string& error) {
    _last_tf_error = error;
  }
}

namespace sony {
  void register_abort_handler(std::function<void(void)> abort_handler) {
    custom_abort_handle = abort_handler;
  }

  void play() {
    std::string error = "Sony internal error!";
    store_error(error);
    custom_abort_handle();
    std::abort();
  }
}
#include <iostream>
#include <string>

#include "sony.h"
#define OPS_EXPORT __attribute__((visibility("default")))

static std::string last_error_message;

extern "C" {
  OPS_EXPORT void _sony_tf_set_error_message(const char* msg) {
    last_error_message = msg;
  }
}

int main(int argc, char** argv) {
  sony::play();
  std::cout << last_error_message << std::endl;
}

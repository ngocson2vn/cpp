#include <iostream>
#include <string>

// sudo apt-get install libgflags2.2 libgflags-dev
#include <gflags/gflags.h>

DEFINE_string(model_dir, "", "model directory");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  printf("Model dir: %s\n", FLAGS_model_dir.c_str());
}

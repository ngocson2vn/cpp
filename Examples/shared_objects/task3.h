#pragma once

#include "task2.h"

#define EXPORT __attribute__((visibility("default")))

extern "C" {
  EXPORT void perform3();
}
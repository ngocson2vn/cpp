#pragma once

#include "task1.h"

#define EXPORT __attribute__((visibility("default")))

extern "C" {
  EXPORT void perform();
}
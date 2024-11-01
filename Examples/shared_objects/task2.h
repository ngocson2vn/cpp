#pragma once

#include "task1.h"

#define EXPORT __attribute__((visibility("default")))

class Task2 {
  public:
    Task2();
    void perform();
};

extern "C" {
  EXPORT void perform();
}
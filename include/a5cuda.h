#pragma once

#include "pch.h"
#include <bits/stdint-uintn.h>

class A5Cuda {
 private:
  uint32_t max_rounds_;
  uint32_t condition_;

 protected:
  void CleanupResources();

 public:
  A5Cuda(uint32_t max_rounds, uint32_t condition);
  ~A5Cuda();
};

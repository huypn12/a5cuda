#pragma once

#include <pch.hh>

namespace a5cuda {
class Advance {
private:
  std::vector<uint64_t> advances_;
  std::vector<uint64_t> rf_table_;

protected:
  uint64_t AdvanceRfLfsr(uint64_t v);
  uint64_t ReverseBits(uint64_t r);

public:
  Advance(unsigned int id, unsigned int size);
  ~Advance();

  const std::vector<uint64_t> &advances() {return advances_;}
  const std::vector<uint64_t> &rf_table() {return rf_table_;}

};
}


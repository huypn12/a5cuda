#pragma once

#include "common.h"

class A5Cuda {
 public:
  // jobpiece_s struct
  // purpose: to handling a single chain, feeding to A5CudaSlice's free slot
  // why a struct to implement???? :))) in the very original version of ati, it
  // was not a class, just separated var but after that they decided to put
  // different chain w/ different adv. together to process in one kernel
  // invocation, so... :)))
  typedef struct {
    uint64_t start_value;
    uint64_t end_value;
    unsigned int start_round;
    unsigned int end_round;
    unsigned int current_round;
    void* context;
    const uint64_t* round_func;
    unsigned int cycles;
    bool idle;
  } JobPiece_s;

  // construct & destruct
  A5Cuda(uint32_t maxRounds, int condition);
  ~A5Cuda();

  // public method for input & output
  int Submit(uint64_t start_value, unsigned int start_round, uint32_t advance,
             void* context);
  int SubmitPartial(uint64_t start_value, unsigned int stop_round,
                    uint32_t advance, void* context);
  bool PopResult(uint64_t&, uint64_t&, void* context);

 private:
  friend class A5CudaSlice;

  // multithreading controller
  bool mRunning;
  // considering of c++11 std::mutex and std::thread
  std::mutex mMutex;
  std::thread* mProcessThread;
  std::vector<A5CudaSlice> mSlices;
  // chain shared common input
  unsigned int mCondition;
  unsigned int mMaxRound;

  // input queues
  deque<uint64_t> mInputStart;
  deque<unsigned int> mInputRoundStart;
  deque<unsigned int> mInputRoundStop;
  deque<void*> mInputContext;
  deque<uint32_t> mInputAdvance;
  // Advance map, to avoid recalculating key chain
  map<uint32_t, class Advance*> mAdvanceMap;

  // output queues
  queue<pair<uint64_t, uint64_t> > mOutput;
  queue<unsigned int> mOutputStartRound;
  queue<unsigned int> mOutputStopRound;
  queue<void*> mOutputContext;

  // access only by A5CudaSlice
  bool PushResult(JobPiece_s*);
  bool PopRequest(JobPiece_s*);
  void Process();
  uint64_t ReverseBits(uint64_t);
};

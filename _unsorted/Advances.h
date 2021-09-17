//
//  Advances.h
//  TestRound
//
//  Created by Doan Trung Tung on 4/14/14.
//  Copyright (c) 2014 Doan Trung Tung. All rights reserved.
//

#ifndef TestRound_Advances_h
#define TestRound_Advances_h

#include <stdint.h>

class Advance {
public:
    Advance(unsigned int id, unsigned int size);
    ~Advance();
    
    const uint64_t* getAdvances() {return mAdvances;}
    const uint32_t* getRFtable() {return mRFtable;}
    
private:
    uint64_t AdvanceRFlfsr(uint64_t v);
    uint64_t ReverseBits(uint64_t r);
    uint64_t* mAdvances;
    uint32_t* mRFtable;
};

#endif

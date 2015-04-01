/***************************************************************
 * A5/1 Chain generator.
 *
 * Copyright 2009. Frank A. Stevenson. All rights reserved.
 *
 * Permission to distribute, modify and copy is granted to the
 * TMTO project, currently hosted at:
 *
 * http://reflextor.com/trac/a51
 *
 * Code may be modifed and used, but not distributed by anyone.
 *
 * Request for alternative licencing may be directed to the author.
 *
 * All (modified) copies of this source must retain this copyright notice.
 *
 *******************************************************************/

/**
 * A5/1 chain generator, CUDA implementation
 * References to "kraken" code of srlabs.de
 * Author: Huy Phung
 * Version: 1.0
 */


#include "A5Cuda.h"
#include "A5CudaSlice.h"


/*
 * Author: Huy Phung
 * Constructor of A5Cuda
 * Set parameter to running thread
 * @param: maxrounds, condition
 * @return: A5Cuda instance
 */
A5Cuda::A5Cuda(uint32_t max_rounds, int condition)
{
    mRunning = true;
    mMaxRound = max_rounds;
    mCondition = condition;
    mProcessThread = new std::thread(&A5Cuda::Process, this);
}

/**
 * Author: Huy Phung
 * Destructor
 * Breaking infinite loop inside Process(), then joining working thread
 */
A5Cuda::~A5Cuda()
{
    mRunning = false;
    mProcessThread->join();
}

/**
 * Author: Huy Phung
 * Rear submit, x->end
 * @param: start value, start round, advance, context
 * @return: size of input queue
 */
int A5Cuda::Submit(
        uint64_t start_value,
        unsigned int start_round,
        uint32_t advance,
        void* context
        )
{
    mMutex.lock();
    mInputStart.push_back(start_value);
    mInputRoundStart.push_back(start_round);
    mInputRoundStop.push_back(mMaxRound);
    mInputContext.push_back(context);
    mInputAdvance.push_back(advance);
    int size = mInputRoundStart.size();

    mMutex.unlock();
    return size;
}

/**
 * Author: Huy Phung
 * Front submit, start->x
 * @param: start value, stop round, advance, context
 * @return: size of input queue
 */
int A5Cuda::SubmitPartial(
        uint64_t start_value,
        unsigned int stop_round,
        uint32_t advance,
        void* context)
{
    mMutex.lock();

    mInputStart.push_front(start_value);
    mInputRoundStart.push_front(0);
    mInputRoundStop.push_front(stop_round);
    mInputContext.push_front(context);
    mInputAdvance.push_front(advance);
    int size = mInputRoundStart.size();

    mMutex.unlock();

    return size;
}

/**
 * Author: Huy Phung
 * Pop a set of (startval, start round, endround, roundfunc), packaging them to Jobpiece_s
 * @param: jobpiece_s
 * @return: false if input queue is empty
 */
bool A5Cuda::PopRequest(JobPiece_s* job)
{

    bool res = false;
    mMutex.lock();
    if (mInputStart.size()>0) {
        res = true;
        job->start_value = mInputStart.front();
        mInputStart.pop_front();
        job->start_round = mInputRoundStart.front();
        mInputRoundStart.pop_front();
        job->end_round = mInputRoundStop.front()-1;
        mInputRoundStop.pop_front();
        job->current_round = job->start_round;
        job->context = mInputContext.front();
        mInputContext.pop_front();
        unsigned int advance = mInputAdvance.front();
        mInputAdvance.pop_front();

        Advance* pAdv;
        map<uint32_t,Advance*>::iterator it = mAdvanceMap.find(advance);
        if (it==mAdvanceMap.end()) {
            pAdv = new Advance(advance,mMaxRound);
            mAdvanceMap[advance]=pAdv;
        } else {
            pAdv = (*it).second;
        }
        job->round_func = pAdv->getAdvances();
        job->cycles = 0;
        job->idle = false;
    }
    mMutex.unlock();
    return res;
}

/**
 * Author: Huy Phung
 * push result from jobpiece to output queue
 * @param: jobpiece
 * @return: true
 */
bool A5Cuda::PushResult(JobPiece_s* job)
{
    mMutex.lock();
    mOutput.push( pair<uint64_t,uint64_t>(job->start_value, job->end_value) );
    mMutex.unlock();
    return true;
}

/**
 * Author: Huy Phung
 * pop a result from result queue
 * @param: uint64_t, uint64_t
 * @return: true if output queue is not empty
 */
bool A5Cuda::PopResult(uint64_t& start_value, uint64_t& stop_value, void* context)//, uint32_t& start_round, uint32_t& stop_round, void** context)
{
    bool res = false;

    mMutex.lock();

    if (mOutput.size() > 0) {
        res = true;
        start_value = mOutput.front().first;
        stop_value = mOutput.front().second;
        mOutput.pop();
    }

    mMutex.unlock();

    return res;
}


/**
 * Author: Huy Phung
 * processing function, invoking each slice to change state
 * TODO: activate multiple slices from multiple devices
 */
#define N_STREAMS 16
void A5Cuda::Process()
{
    //A5CudaSlice* slice = new A5CudaSlice(this, 0, mCondition, mMaxRound);
    A5CudaSlice* slices[N_STREAMS];
    for (int i= 0; i < N_STREAMS; i++) {
        slices[i] = new A5CudaSlice(this, 0, mCondition, mMaxRound);
    }
    /**
     * TODO: each device has exactly 16 streams, scan over all stream to push new job
     */
    for (;;)
    {
        mMutex.lock();
        int available = mInputStart.size();
        mMutex.unlock();

        if (available == 0) {
            usleep(10);
        }
        for (int i = 0; i < N_STREAMS; i++) {
            (*slices[i]).tick();
        }

        if (!mRunning)
            break;
    }
    for (int i=0; i < N_STREAMS; i++) {
        delete slices[i];
    }
}

extern "C" {
    static class A5Cuda* a5Instance = 0;

    bool DLL_PUBLIC A5CudaInit(int max_rounds, int condition) {
        if (a5Instance) {
            return false;
        }
        a5Instance = new A5Cuda(max_rounds, condition);
        return true;
    }

    int DLL_PUBLIC A5CudaSubmit(uint64_t start_value,
            int32_t start_round, uint32_t advance,
            void* context) {
        if (a5Instance) {
            return a5Instance->Submit(start_value, start_round,
                    advance, context);
        }
        return -1;
    }

    int DLL_PUBLIC A5CudaSubmitPartial(uint64_t start_value,
            int32_t stop_round, uint32_t advance,
            void* context) {
        if (a5Instance) {
            return a5Instance->SubmitPartial(start_value, stop_round,
                    advance, context);
        }
        return -1;
    }

    int DLL_PUBLIC A5CudaPopResult(uint64_t& start_value, uint64_t& stop_value,
            void* context) {
        if (a5Instance) {
            return a5Instance->PopResult(start_value, stop_value, context);
        }
        return -1;
    }

    void DLL_PUBLIC A5CudaShutdown() {
        delete a5Instance;
        a5Instance = NULL;
    }
}

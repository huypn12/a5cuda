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
    processThread = new boost::thread(boost::bind(&A5Cuda::Process, this));
}

/**
 * Author: Huy Phung
 * Destructor
 * Breaking infinite loop inside Process(), then joining working thread
 */
A5Cuda::~A5Cuda()
{
    mRunning = false;
    processThread->join();
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
        job->end_round = mInputRoundStop.front();
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
bool A5Cuda::PopResult(uint64_t& start_value, uint64_t& stop_value)//, uint32_t& start_round, uint32_t& stop_round, void** context)
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
void A5Cuda::Process()
{
    A5CudaSlice* slice = new A5CudaSlice(this, 0, mCondition, mMaxRound);

    /**
     * TODO:
     */
    printf("aoeuaoue\n", "aoeuaoeu");
    for (;;)
    {
        mMutex.lock();
        int available = mInputStart.size();
        mMutex.unlock();

        if (available == 0) {
            usleep(10);
        }

        slice->tick();

        if (!mRunning)
            break;
    }
    delete slice;
}


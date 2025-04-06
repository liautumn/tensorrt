#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include "Timer.h"
#include "Logger.h"

namespace trt_timer {
    Timer::Timer() {
        checkRuntime(cudaEventCreate((cudaEvent_t *) &start_));
        checkRuntime(cudaEventCreate((cudaEvent_t *) &stop_));
    }

    Timer::~Timer() {
        checkRuntime(cudaEventDestroy((cudaEvent_t) start_));
        checkRuntime(cudaEventDestroy((cudaEvent_t) stop_));
    }

    void Timer::start(void *stream) {
        stream_ = stream;
        checkRuntime(cudaEventRecord((cudaEvent_t) start_, (cudaStream_t) stream_));
    }

    float Timer::stop(const char *prefix, bool print) {
        checkRuntime(cudaEventRecord((cudaEvent_t) stop_, (cudaStream_t) stream_));
        checkRuntime(cudaEventSynchronize((cudaEvent_t) stop_));

        float latency = 0;
        checkRuntime(cudaEventElapsedTime(&latency, (cudaEvent_t) start_, (cudaEvent_t) stop_));

        if (print) {
            printf("[%s]: %.5f ms\n", prefix, latency);
        }
        return latency;
    }
}

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include "Timer.h"
#include "Logger.h"

namespace trt_timer {
    Timer::Timer() {
        checkRuntime(cudaEventCreate(reinterpret_cast<cudaEvent_t *>(&start_)));
        checkRuntime(cudaEventCreate(reinterpret_cast<cudaEvent_t *>(&stop_)));
    }

    Timer::~Timer() {
        checkRuntime(cudaEventDestroy(static_cast<cudaEvent_t>(start_)));
        checkRuntime(cudaEventDestroy(static_cast<cudaEvent_t>(stop_)));
    }

    void Timer::start(void *stream) {
        stream_ = stream;
        checkRuntime(cudaEventRecord(static_cast<cudaEvent_t>(start_), static_cast<cudaStream_t>(stream_)));
    }

    float Timer::stop(const char *prefix, bool print) {
        checkRuntime(cudaEventRecord(static_cast<cudaEvent_t>(stop_), static_cast<cudaStream_t>(stream_)));
        checkRuntime(cudaEventSynchronize(static_cast<cudaEvent_t>(stop_)));

        float latency = 0;
        checkRuntime(cudaEventElapsedTime(&latency, static_cast<cudaEvent_t>(start_), static_cast<cudaEvent_t>(stop_)));

        if (print) {
            printf("[%s]: %.5f ms\n", prefix, latency);
        }
        return latency;
    }
}

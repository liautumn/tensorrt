#ifndef TIMER_H
#define TIMER_H

namespace trt_timer {
    class Timer {
    public:
        Timer();

        virtual ~Timer();

        void start(void *stream = nullptr);

        float stop(const char *prefix = "Timer", bool print = true);

    private:
        void *start_, *stop_;
        void *stream_;
    };
}

#endif //TIMER_H

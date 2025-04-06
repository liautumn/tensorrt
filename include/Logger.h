#ifndef LOGGER_H
#define LOGGER_H

#define INFO(...) trt_log::_log_func(__FILE__, __LINE__, __VA_ARGS__)
#define checkRuntime(call)                                                                      \
        do {                                                                                    \
            auto ___call__ret_code__ = (call);                                                  \
            if (___call__ret_code__ != cudaSuccess) {                                           \
                INFO("CUDA Runtime error? %s # %s, code = %s [ %d ]", #call,                    \
                cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
                ___call__ret_code__);                                                           \
                abort();                                                                        \
            }                                                                                   \
        } while (0)

#define checkKernel(...)                            \
        do {                                        \
            { (__VA_ARGS__); }                      \
            checkRuntime(cudaPeekAtLastError());    \
        } while (0)

#define Assert(op)                              \
        do {                                    \
            bool cond = !(!(op));               \
            if (!cond) {                        \
                INFO("Assert failed, " #op);    \
                abort();                        \
            }                                   \
        } while (0)

#define Assertf(op, ...)                                        \
        do {                                                    \
            bool cond = !(!(op));                               \
            if (!cond) {                                        \
                INFO("Assert failed, " #op " : " __VA_ARGS__);  \
                abort();                                        \
            }                                                   \
        } while (0)

namespace trt_log {
    using namespace std;

    void _log_func(const char *file, int line, const char *fmt, ...);
}


#endif //LOGGER_H

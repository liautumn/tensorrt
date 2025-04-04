#ifndef INFER_H_
#define INFER_H_

#include <memory>
#include <string>
#include <vector>

using namespace std;

#define INFO(...) trt::__log_func(__FILE__, __LINE__, __VA_ARGS__)
#define checkRuntime(call)                                                                          \
        do {                                                                                        \
            auto ___call__ret_code__ = (call);                                                      \
                if (___call__ret_code__ != cudaSuccess) {                                           \
                INFO("CUDA Runtime error? %s # %s, code = %s [ %d ]", #call,                       \
                cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__),     \
                ___call__ret_code__);                                                               \
                abort();                                                                            \
            }                                                                                       \
        } while (0)
#define checkKernel(...)                            \
        do {                                        \
            { (__VA_ARGS__); }                      \
            checkRuntime(cudaPeekAtLastError());    \
        } while (0)

namespace trt {
    void __log_func(const char *file, int line, const char *fmt, ...);

    enum class DType : int {
        FLOAT = 0, HALF = 1, INT8 = 2, INT32 = 3, BOOL = 4, UINT8 = 5
    };

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

    class BaseMemory {
    public:
        BaseMemory() = default;

        BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

        virtual ~BaseMemory();

        virtual void *gpu_realloc(size_t bytes);

        virtual void *cpu_realloc(size_t bytes);

        void release_gpu();

        void release_cpu();

        void release();

        inline bool owner_gpu() const { return owner_gpu_; }

        inline bool owner_cpu() const { return owner_cpu_; }

        inline size_t cpu_bytes() const { return cpu_bytes_; }

        inline size_t gpu_bytes() const { return gpu_bytes_; }

        virtual inline void *get_gpu() const { return gpu_; }

        virtual inline void *get_cpu() const { return cpu_; }

        void reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

    protected:
        void *cpu_ = nullptr;
        size_t cpu_bytes_ = 0, cpu_capacity_ = 0;
        bool owner_cpu_ = true;

        void *gpu_ = nullptr;
        size_t gpu_bytes_ = 0, gpu_capacity_ = 0;
        bool owner_gpu_ = true;
    };

    template<typename _DT>
    class Memory : public BaseMemory {
    public:
        Memory() = default;

        Memory(const Memory &other) = delete;

        Memory &operator=(const Memory &other) = delete;

        virtual _DT *gpu(size_t size) { return (_DT *) BaseMemory::gpu_realloc(size * sizeof(_DT)); }

        virtual _DT *cpu(size_t size) { return (_DT *) BaseMemory::cpu_realloc(size * sizeof(_DT)); }

        inline size_t cpu_size() const { return cpu_bytes_ / sizeof(_DT); }

        inline size_t gpu_size() const { return gpu_bytes_ / sizeof(_DT); }

        virtual inline _DT *gpu() const { return (_DT *) gpu_; }

        virtual inline _DT *cpu() const { return (_DT *) cpu_; }
    };

    class Infer {
    public:
        virtual bool forward(const vector<void *> &bindings, void *stream = nullptr,
                             void *input_consum_event = nullptr) = 0;

        virtual int index(const string &name) = 0;

        virtual vector<int> run_dims(const string &name) = 0;

        virtual vector<int> run_dims(int ibinding) = 0;

        virtual vector<int> static_dims(const string &name) = 0;

        virtual vector<int> static_dims(int ibinding) = 0;

        virtual int numel(const string &name) = 0;

        virtual int numel(int ibinding) = 0;

        virtual int num_bindings() = 0;

        virtual bool is_input(int ibinding) = 0;

        virtual bool set_run_dims(const string &name, const vector<int> &dims) = 0;

        virtual bool set_run_dims(int ibinding, const vector<int> &dims) = 0;

        virtual DType dtype(const string &name) = 0;

        virtual DType dtype(int ibinding) = 0;

        virtual bool has_dynamic_dim() = 0;

        virtual void print() = 0;
    };

    shared_ptr<Infer> load(const string &file);

    string format_shape(const vector<int> &shape);
} // namespace trt

#endif  // INFER_H_

#ifndef MEMORY_H
#define MEMORY_H

#include "BaseMemory.h"

namespace trt_memory {
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
}

#endif //MEMORY_H

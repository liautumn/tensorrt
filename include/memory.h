#ifndef MEMORY_H
#define MEMORY_H

#include "base_memory.h"

namespace trt_memory {
    template<typename DT>
    class Memory : public BaseMemory {
    public:
        Memory() = default;

        Memory(const Memory &other) = delete;

        Memory &operator=(const Memory &other) = delete;

        virtual DT *gpu(size_t size) { return static_cast<DT *>(BaseMemory::gpu_realloc(size * sizeof(DT))); }

        virtual DT *cpu(size_t size) { return static_cast<DT *>(BaseMemory::cpu_realloc(size * sizeof(DT))); }

        size_t cpu_size() const { return cpu_bytes_ / sizeof(DT); }

        size_t gpu_size() const { return gpu_bytes_ / sizeof(DT); }

        virtual DT *gpu() const { return static_cast<DT *>(gpu_); }

        virtual DT *cpu() const { return static_cast<DT *>(cpu_); }
    };
}

#endif //MEMORY_H

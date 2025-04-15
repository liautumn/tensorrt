#ifndef BASEMEMORY_H
#define BASEMEMORY_H

namespace trt_memory {
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

        bool owner_gpu() const { return owner_gpu_; }

        bool owner_cpu() const { return owner_cpu_; }

        size_t cpu_bytes() const { return cpu_bytes_; }

        size_t gpu_bytes() const { return gpu_bytes_; }

        virtual void *get_gpu() const { return gpu_; }

        virtual void *get_cpu() const { return cpu_; }

        void reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

    protected:
        void *cpu_ = nullptr;
        size_t cpu_bytes_ = 0, cpu_capacity_ = 0;
        bool owner_cpu_ = true;

        void *gpu_ = nullptr;
        size_t gpu_bytes_ = 0, gpu_capacity_ = 0;
        bool owner_gpu_ = true;
    };
}

#endif //BASEMEMORY_H

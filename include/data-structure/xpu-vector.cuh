#pragma once
#include <cassert>
#include <cstdio>

namespace xpu {

    enum class mem_policy { host_only, device_only, cross_platform };

    template <typename T>
    class vector {
        T *cpu_data = 0, *gpu_data = 0;
        size_t size_ = 0;
        mem_policy mp;
    public:
        T& operator[](size_t idx) { return cpu_data[idx]; }
        const T& operator[](size_t idx) const { return cpu_data[idx]; }
        void* p() {
            assert(mp != mem_policy::host_only);
            return gpu_data;
        }

        vector() : size_(0) {}
        vector(size_t size_, mem_policy mp = mem_policy::host_only) : size_(size_), mp(mp) 
        {
            switch (mp) {
                case mem_policy::host_only: {
                    cpu_data = new T[size_];
                } break;
                case mem_policy::device_only: {
                    cudaMalloc((void**)&gpu_data, size_ * sizeof(T));
                } break;
                case mem_policy::cross_platform: {
                    cpu_data = new T[size_];
                    cudaMalloc((void**)&gpu_data, size_ * sizeof(T));
                } break;
            }
        }
        vector(size_t cpu_size, size_t gpu_size) : size_(cpu_size), mp(mem_policy::cross_platform) 
        {
            cpu_data = new T[cpu_size];
            cudaMalloc((void**)&gpu_data, gpu_size * sizeof(T));
        }

        void allocate(size_t size__, mem_policy mp_ = mem_policy::host_only) 
        {
            size_ = size__, mp = mp_;

            switch (mp) {
                case mem_policy::host_only: {
                    cpu_data = new T[size_];
                } break;
                case mem_policy::device_only: {
                    cudaMalloc((void**)&gpu_data, size_ * sizeof(T));
                } break;
                case mem_policy::cross_platform: {
                    cpu_data = new T[size_];
                    if (size_ >= 1) cpu_data[0] = cpu_data[1];
                    cudaMalloc((void**)&gpu_data, size_ * sizeof(T));
                } break;
            }
        }


        ~vector() 
        {
            switch (mp) {
                case mem_policy::host_only: {
                    delete[] cpu_data;
                } break;
                case mem_policy::device_only: {
                    cudaFree(gpu_data);
                } break;
                case mem_policy::cross_platform: {
                    delete[] cpu_data;
                    cudaFree(gpu_data);
                } break;
            }
        }

        void store() {
            assert(mp == mem_policy::cross_platform);
            cudaMemcpy(gpu_data, cpu_data, size_ * sizeof(T), cudaMemcpyHostToDevice);
        }

        void load() {
            assert(mp == mem_policy::cross_platform);
            cudaMemcpy(cpu_data, gpu_data, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        }

        void load(size_t length) {
            assert(mp == mem_policy::cross_platform);
            cudaMemcpy(cpu_data, gpu_data, length * sizeof(T), cudaMemcpyDeviceToHost);
        }

        void load(void *src, size_t length) {
            assert(mp != mem_policy::device_only);
            cudaMemcpy(cpu_data, src, length * sizeof(T), cudaMemcpyDeviceToHost);
        }

        void clear() {
            assert(mp != mem_policy::host_only);
            cudaMemset(gpu_data, 0, size_ * sizeof(T));
            cudaDeviceSynchronize();
        }

    };

}
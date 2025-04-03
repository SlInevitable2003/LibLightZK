#include "xpu-vector.cuh"

namespace xpu {
    template <typename T, int dev_cnt>
    class multigpu_vector {
        T *cpu_data = 0, *gpu_data[dev_cnt] = 0;
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
                    #pragma omp parallel for
                    for (int dev = 0; dev < dev_cnt; dev++) {
                        cudaSetDevice(dev);
                        cudaMalloc((void**)&gpu_data, size_ * sizeof(T));
                    }
                } break;
                case mem_policy::cross_platform: {
                    cpu_data = new T[size_];
                    #pragma omp parallel for
                    for (int dev = 0; dev < dev_cnt; dev++) {
                        cudaSetDevice(dev);
                        cudaMalloc((void**)&gpu_data, size_ * sizeof(T));
                    }
                } break;
            }
        }

        void allocate(size_t size__, mem_policy mp_ = mem_policy::host_only) 
        {
            size_ = size__, mp = mp_;

            switch (mp) {
                case mem_policy::host_only: {
                    cpu_data = new T[size_];
                } break;
                case mem_policy::device_only: {
                    #pragma omp parallel for
                    for (int dev = 0; dev < dev_cnt; dev++) {
                        cudaSetDevice(dev);
                        cudaMalloc((void**)&gpu_data[dev], size_ * sizeof(T));
                    }
                } break;
                case mem_policy::cross_platform: {
                    cpu_data = new T[size_];
                    #pragma omp parallel for
                    for (int dev = 0; dev < dev_cnt; dev++) {
                        cudaSetDevice(dev);
                        cudaMalloc((void**)&gpu_data[dev], size_ * sizeof(T));
                    }
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
                    #pragma omp parallel for
                    for (int dev = 0; dev < dev_cnt; dev++) {
                        cudaSetDevice(dev);
                        cudaFree(gpu_data[dev]);
                    }
                } break;
                case mem_policy::cross_platform: {
                    delete[] cpu_data;
                    #pragma omp parallel for
                    for (int dev = 0; dev < dev_cnt; dev++) {
                        cudaSetDevice(dev);
                        cudaFree(gpu_data[dev]);
                    }
                } break;
            }
        }

        void store() {
            assert(mp == mem_policy::cross_platform);
            #pragma omp parallel for
            for (int dev = 0; dev < dev_cnt; dev++) {
                cudaSetDevice(dev);
                cudaMemcpy(gpu_data[dev], cpu_data, size_ * sizeof(T), cudaMemcpyHostToDevice);
            }
        }

        void load(int dev) {
            assert(mp == mem_policy::cross_platform);
            cudaMemcpy(cpu_data, gpu_data[dev], size_ * sizeof(T), cudaMemcpyDeviceToHost);
        }

        void clear() {
            assert(mp != mem_policy::host_only);
            #pragma omp parallel for
            for (int dev = 0; dev < dev_cnt; dev++) {
                cudaSetDevice(dev);
                cudaMemset(gpu_data[dev], 0, size_ * sizeof(T));
                cudaDeviceSynchronize();
            }
        }

    };
}
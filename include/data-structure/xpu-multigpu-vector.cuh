#include "xpu-vector.cuh"
#include "device-utils.cuh"

namespace xpu {
    template <typename T, int dev_cnt>
    class multigpu_vector {
        T *cpu_data = 0, *gpu_data[dev_cnt] = { 0 };
        size_t size_ = 0;
        mem_policy mp;
    public:
        T& operator[](size_t idx) { return cpu_data[idx]; }
        const T& operator[](size_t idx) const { return cpu_data[idx]; }
        void* p(int dev) {
            assert(mp != mem_policy::host_only);
            return gpu_data[dev];
        }

        multigpu_vector() : size_(0) {}
        multigpu_vector(size_t size_, mem_policy mp = mem_policy::host_only) : size_(size_), mp(mp) 
        {
            switch (mp) {
                case mem_policy::host_only: {
                    cpu_data = new T[size_];
                } break;
                case mem_policy::device_only: {
                    #pragma omp parallel for
                    for (int dev = 0; dev < dev_cnt; dev++) {
                        cudaSetDevice(dev);
                        cudaMalloc((void**)&gpu_data[dev], size_ * sizeof(T));
                        cudaDeviceSynchronize();
                    }
                } break;
                case mem_policy::cross_platform: {
                    cpu_data = new T[size_];
                    #pragma omp parallel for
                    for (int dev = 0; dev < dev_cnt; dev++) {
                        cudaSetDevice(dev);
                        cudaMalloc((void**)&gpu_data[dev], size_ * sizeof(T));
                        cudaDeviceSynchronize();
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
                        cudaDeviceSynchronize();
                    }
                } break;
                case mem_policy::cross_platform: {
                    cpu_data = new T[size_];
                    #pragma omp parallel for
                    for (int dev = 0; dev < dev_cnt; dev++) {
                        cudaSetDevice(dev);
                        cudaMalloc((void**)&gpu_data[dev], size_ * sizeof(T));
                        cudaDeviceSynchronize();
                    }
                } break;
            }
        }


        ~multigpu_vector() 
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
                cudaDeviceSynchronize();
            }
        }

        void load(int dev) {
            assert(mp == mem_policy::cross_platform && dev < dev_cnt);
            cudaSetDevice(dev);
            cudaMemcpy(cpu_data, gpu_data[dev], size_ * sizeof(T), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaSetDevice(0);
        }

        void move(int dest, int src) {
            assert(mp != mem_policy::host_only && dest < dev_cnt && src < dev_cnt);
            int canAccess;
            cudaDeviceCanAccessPeer(&canAccess, src, dest);
            if (canAccess) {
                cudaSetDevice(src);
                cudaDeviceEnablePeerAccess(dest, src);
                cudaMemcpyPeer(gpu_data[dest], dest, gpu_data[src], src, size_ * sizeof(T));
            } else {
                T *host_buf = new T[size_];
                cudaSetDevice(src);
                cudaMemcpy(host_buf, gpu_data[src], size_ * sizeof(T), cudaMemcpyDeviceToHost);
                cudaSetDevice(src);
                cudaMemcpy(gpu_data[dest], host_buf, size_ * sizeof(T), cudaMemcpyHostToDevice);
                delete[] host_buf;
            }
            cudaDeviceSynchronize();

            cudaSetDevice(0);
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
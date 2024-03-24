#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#define VALUES_PER_WORK_ITEM 64

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    // TODO: implement on OpenCL
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_global_atomic");

        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u result_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        result_gpu.resizeN(1);

        kernel.compile();
        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            unsigned int sum = 0;
            result_gpu.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu,
                        result_gpu,
                        n);
            result_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU global atomic sum should be consistent!");
            t.nextLap();
        }
        size_t gflops = 1000 * 1000 * 1000;
        std::cout << "GPU global atomic: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU global atomic: " << n / t.lapAvg() / gflops << " GFlops" << std::endl;
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_loop");

        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u result_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        result_gpu.resizeN(1);

        kernel.compile();
        unsigned int workGroupSize = 128;
        unsigned int global_work_size = ((n + VALUES_PER_WORK_ITEM - 1) / VALUES_PER_WORK_ITEM
                + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            unsigned int sum = 0;
            result_gpu.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu,
                        result_gpu,
                        n);
            result_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU loop sum should be consistent!");
            t.nextLap();
        }
        size_t gflops = 1000 * 1000 * 1000;
        std::cout << "GPU loop: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU loop: " << n / t.lapAvg() / gflops << " GFlops" << std::endl;
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_loop_coalesced");

        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u result_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        result_gpu.resizeN(1);

        kernel.compile();
        unsigned int workGroupSize = 128;
        unsigned int global_work_size = ((n + VALUES_PER_WORK_ITEM - 1) / VALUES_PER_WORK_ITEM
                                         + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            unsigned int sum = 0;
            result_gpu.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu,
                        result_gpu,
                        n);
            result_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU coalesced loop sum should be consistent!");
            t.nextLap();
        }
        size_t gflops = 1000 * 1000 * 1000;
        std::cout << "GPU coalesced loop: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU coalesced loop: " << n / t.lapAvg() / gflops << " GFlops" << std::endl;
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_local_memory");

        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u result_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        result_gpu.resizeN(1);

        kernel.compile();
        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            unsigned int sum = 0;
            result_gpu.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu,
                        result_gpu,
                        n);
            result_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU with local memory sum should be consistent!");
            t.nextLap();
        }
        size_t gflops = 1000 * 1000 * 1000;
        std::cout << "GPU with local memory: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU with local memory: " << n / t.lapAvg() / gflops << " GFlops" << std::endl;
    }
}

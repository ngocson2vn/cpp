#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/functional.h>


#include <sys/time.h>

using Vec = std::vector<float>;
constexpr size_t TEST_SIZE = 100'000'000;
constexpr size_t SAMPLE_IDX = 999'999;

// vvv start vvv
float compute(Vec& va, Vec& vb, Vec& vc, Vec& vr, Vec& vd) {
	for (size_t k = 0; k < 50; ++k) {
		for (size_t i = 0; i < TEST_SIZE; ++i) {
			vd[i] += (va[i] - vb[i]) * vc[i] * vr[(i * 37+k) % TEST_SIZE];
		}
	}
	std::sort(vd.begin(), vd.end(), std::greater<float>());
	return vd[SAMPLE_IDX];
}
// ^^^ end ^^^

template <size_t TEST_SIZE, size_t SAMPLE_IDX>
__global__ void compute_kernel(
	const float* __restrict__ va,
	const float* __restrict__ vb,
	const float* __restrict__ vc,
	const float* __restrict__ vr,
	float* __restrict__ vd
) {
	const int TOTAL_THREADS = gridDim.x * blockDim.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < TEST_SIZE) {
		float va_v = va[i];
		float vb_v = vb[i];
		float vc_v = vc[i];

		float acc = 0;
		#pragma unroll 50
		for (size_t k = 0; k < 50; ++k) {
			acc += (va_v - vb_v) * vc_v * vr[(i * 37+k) % TEST_SIZE];
		}

		vd[i] = acc;

		i += TOTAL_THREADS;
	}
}

float gpu_compute(Vec& va, Vec& vb, Vec& vc, Vec& vr, Vec& vd) {
	const size_t N = va.size();
	int threads = 256;
	int blocks = 1;
	cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, compute_kernel<TEST_SIZE, SAMPLE_IDX>, 0, 0);

	threads = N < threads ? N : threads;
	int heuBlocks = (N + threads - 1) / threads;
	blocks = N < blocks * threads ? heuBlocks : blocks;
	printf("blocks = %d threads = %d\n", blocks, threads);
	float* dev_va_ptr = nullptr;
	float* dev_vb_ptr = nullptr;
	float* dev_vc_ptr = nullptr;
	float* dev_vr_ptr = nullptr;
	float* dev_vd_ptr = nullptr;
	size_t numBytes = N * sizeof(float);

	// Input
	cudaMalloc(&dev_va_ptr, numBytes);
	cudaMalloc(&dev_vb_ptr, numBytes);
	cudaMalloc(&dev_vc_ptr, numBytes);
	cudaMalloc(&dev_vr_ptr, numBytes);
	cudaMemcpy(dev_va_ptr, va.data(), numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vb_ptr, vb.data(), numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vc_ptr, vc.data(), numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vr_ptr, vr.data(), numBytes, cudaMemcpyHostToDevice);

	// Output
	cudaMalloc(&dev_vd_ptr, numBytes);
	
	compute_kernel<TEST_SIZE, SAMPLE_IDX><<<blocks, threads>>>(dev_va_ptr, dev_vb_ptr, dev_vc_ptr, dev_vr_ptr, dev_vd_ptr);
	cudaDeviceSynchronize();

	// Sort later
	thrust::device_ptr<float> dev_ptr(dev_vd_ptr);
	thrust::sort(dev_ptr, dev_ptr + N, thrust::greater<float>());
	thrust::host_vector<float> host_vec(N, 0);
	cudaMemcpy(host_vec.data(), dev_vd_ptr, numBytes, cudaMemcpyDeviceToHost);

	return host_vec[SAMPLE_IDX];
}

int main(void) {
    Vec va(TEST_SIZE);
    Vec vb(TEST_SIZE);
    Vec vc(TEST_SIZE);
    Vec vr(TEST_SIZE);
    Vec vd(TEST_SIZE, 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-5, 5);
    std::uniform_real_distribution<float> dis2(0.8, 1.2);
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        va[i] = dis(gen);
        vb[i] = dis(gen);
        vc[i] = dis(gen);
        vr[i] = dis(gen);
    }
    printf("Initialized, starting calculation\n");
    struct timeval pos_1, pos_2;
    gettimeofday(&pos_1, NULL);

		float gpu_k = gpu_compute(va, vb, vc, vr, vd);
    std::cout << "GPU result: " << gpu_k << std::endl;
    gettimeofday(&pos_2, NULL);
    printf("calculation cost %.6f seconds\n", (1'000'000 * (pos_2.tv_sec - pos_1.tv_sec) + pos_2.tv_usec - pos_1.tv_usec) / 1.e6);

		float cpu_k = compute(va, vb, vc, vr, vd);
		std::cout << "CPU result: " << cpu_k << std::endl;

    return 0;
}

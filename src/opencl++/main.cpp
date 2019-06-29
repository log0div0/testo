
#include <iostream>
#include <chrono>
#include "context.hpp"

void backtrace(std::ostream& stream, const std::exception& error, size_t n) {
	stream << n << ". " << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const std::exception& error) {
		stream << std::endl;
		backtrace(stream, error, n + 1);
	} catch(...) {
		stream << std::endl;
		stream << n << ". " << "[Unknown exception type]";
	}
}

std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error, 1);
	return stream;
}

std::tuple<cl::Platform, cl::Device> get_platform_and_device() {
	for (cl::Platform platform: cl::Platform::ids()) {
		for (cl::Device device: platform.device_ids(CL_DEVICE_TYPE_GPU)) {
			return {platform, device};
		}
	}
	throw std::runtime_error("GPU not found");
}

const char *kernelstring = R"(
	__kernel void myGEMM1(const size_t M, const size_t N, const size_t K,
							const __global float* A, const __global float* B, __global float* C)
	{
		const size_t globalRow = get_global_id(0);
		const size_t globalCol = get_global_id(1);
		float acc = 0.0f;
		for (size_t k=0; k<K; k++) {
			acc += A[k*M + globalRow] * B[globalCol*K + k];
		}
		C[globalCol*M + globalRow] = acc;
	}
)";

void main() {
	try {
		auto [platform, device] = get_platform_and_device();

		std::cout << "Using device " << device.name() << std::endl;

		size_t K = 4096;
		size_t M = 4096;
		size_t N = 4096;

		std::vector<float> A(M*K);
		std::vector<float> B(K*N);
		std::vector<float> C(M*N);

		cl::Context context(platform, {device});
		cl::CommandQueue queue = context.createCommandQueue(device);
		cl::Program program = context.createProgram({kernelstring});
		try {
			program.build({device});
		} catch (const std::exception& error) {
			std::cout << program.build_log(device) << std::endl;
			return;
		}
		cl::Mem bufA = context.createBuffer(CL_MEM_READ_ONLY,  M*K*sizeof(float));
		cl::Mem bufB = context.createBuffer(CL_MEM_READ_ONLY,  K*N*sizeof(float));
		cl::Mem bufC = context.createBuffer(CL_MEM_READ_WRITE, M*N*sizeof(float));

		cl::Kernel kernel = program.createKernel("myGEMM1");

		auto start = std::chrono::high_resolution_clock::now();

		kernel.setArg(0, sizeof(M), &M);
		kernel.setArg(1, sizeof(N), &N);
		kernel.setArg(2, sizeof(K), &K);
		kernel.setArg(3, sizeof(bufA), &bufA);
		kernel.setArg(4, sizeof(bufB), &bufB);
		kernel.setArg(5, sizeof(bufC), &bufC);

		cl::wait({
			queue.readBuffer(bufC, 0, M*N*sizeof(float), C.data(), {
				queue.execute(kernel, {M, N}, {
					queue.writeBuffer(bufA, 0, M*K*sizeof(float), A.data()),
					queue.writeBuffer(bufB, 0, K*N*sizeof(float), B.data()),
					queue.writeBuffer(bufC, 0, M*N*sizeof(float), C.data())
				})
			})
		});

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end - start;
		std::cout << time.count() << " seconds" << std::endl;
	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
}

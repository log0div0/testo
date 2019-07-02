
#include <iostream>
#include <chrono>
#include <sstream>
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

void gemm_nn(int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float *C, int ldc)
{
	int i,j,k;
	for(i = 0; i < M; ++i){
		for(k = 0; k < K; ++k){
			float A_PART = ALPHA*A[i*lda+k];
			for(j = 0; j < N; ++j){
				C[i*ldc+j] += A_PART*B[k*ldb+j];
			}
		}
	}
}

void gemm_nt(int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float *C, int ldc)
{
	int i,j,k;
	for(i = 0; i < M; ++i){
		for(j = 0; j < N; ++j){
			float sum = 0;
			for(k = 0; k < K; ++k){
				sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
			}
			C[i*ldc+j] += sum;
		}
	}
}

void gemm_tn(int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float *C, int ldc)
{
	int i,j,k;
	for(i = 0; i < M; ++i){
		for(k = 0; k < K; ++k){
			float A_PART = ALPHA*A[k*lda+i];
			for(j = 0; j < N; ++j){
				C[i*ldc+j] += A_PART*B[k*ldb+j];
			}
		}
	}
}

void gemm_tt(int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float *C, int ldc)
{
	int i,j,k;
	for(i = 0; i < M; ++i){
		for(j = 0; j < N; ++j){
			float sum = 0;
			for(k = 0; k < K; ++k){
				sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
			}
			C[i*ldc+j] += sum;
		}
	}
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float BETA,
		float *C, int ldc)
{
	//printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
	int i, j;
	for(i = 0; i < M; ++i){
		for(j = 0; j < N; ++j){
			C[i*ldc + j] *= BETA;
		}
	}
	if(!TA && !TB)
		gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
	else if(TA && !TB)
		gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
	else if(!TA && TB)
		gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
	else
		gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

const char *kernelstring = R"(

__kernel void gemm(const int M, const int N, const int K,
		const float ALPHA,
		const __global float *A, const int lda,
		const __global float *B, const int ldb,
		const float BETA,
		__global float *C, const int ldc)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);

	float acc = 0.0f;
	for (int k = 0; k < K; ++k) {
#if TA == 0
		int A_index = i*lda+k;
#else
		int A_index = k*lda+i;
#endif
#if TB == 0
		int B_index = k*ldb+j;
#else
		int B_index = j*ldb+k;
#endif
		acc += ALPHA*A[A_index]*B[B_index];
	}
	C[i*ldc+j] *= BETA;
	C[i*ldc+j] += acc;
}

)";

struct Dummy {

	Dummy() {
		bool device_found = false;
		for (cl::Platform platform: cl::Platform::ids()) {
			for (cl::Device device: platform.device_ids(CL_DEVICE_TYPE_GPU)) {
				this->platform = platform;
				this->device = device;
				device_found = true;
				break;
			}
		}
		if (!device_found) {
			throw std::runtime_error("GPU not found");
		}

		context = cl::Context(platform, {device});
		for (auto TA: {false, true}) {
			for (auto TB: {false, true}) {
				auto& program = programs[TA][TB];
				program = context.createProgram({kernelstring});
				std::stringstream options;
				options
					<< "-D TA=" << (int)TA
					<< " -D TB=" << (int)TB;
				try {
					program.build({device}, options.str());
				} catch (const std::exception& error) {
					std::cout << program.build_log(device) << std::endl;
					throw;
				}
			}
		}
	}

	void gemm(bool TA, bool TB,
			int M, int N, int K,
			float ALPHA,
			float *A, int lda,
			float *B, int ldb,
			float BETA,
			float *C, int ldc)
	{
		cl::Mem bufA = context.createBuffer(CL_MEM_READ_ONLY,  M*K*sizeof(float));
		cl::Mem bufB = context.createBuffer(CL_MEM_READ_ONLY,  K*N*sizeof(float));
		cl::Mem bufC = context.createBuffer(CL_MEM_READ_WRITE, M*N*sizeof(float));

		cl::Kernel kernel = programs[TA][TB].createKernel("gemm");

		kernel.setArg(0, sizeof(M), &M);
		kernel.setArg(1, sizeof(N), &N);
		kernel.setArg(2, sizeof(K), &K);
		kernel.setArg(3, sizeof(ALPHA), &ALPHA);
		kernel.setArg(4, sizeof(bufA), &bufA);
		kernel.setArg(5, sizeof(lda), &lda);
		kernel.setArg(6, sizeof(bufB), &bufB);
		kernel.setArg(7, sizeof(ldb), &ldb);
		kernel.setArg(8, sizeof(BETA), &BETA);
		kernel.setArg(9, sizeof(bufC), &bufC);
		kernel.setArg(10, sizeof(ldc), &ldc);

		cl::CommandQueue queue = context.createCommandQueue(device);
		cl::wait({
			queue.readBuffer(bufC, 0, M*N*sizeof(float), C, {
				queue.execute(kernel, {(size_t)M, (size_t)N}, {TS, TS}, {
					queue.writeBuffer(bufA, 0, M*K*sizeof(float), A),
					queue.writeBuffer(bufB, 0, K*N*sizeof(float), B),
					queue.writeBuffer(bufC, 0, M*N*sizeof(float), C)
				})
			})
		});
	}

	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::Program programs[2][2];
	size_t TS = 16;
};

std::vector<float> random_matrix(int rows, int cols)
{
    std::vector<float> m(rows*cols);
    for(int i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void test_opencl_accuracy(int TA, int TB, int m, int k, int n)
{
	srand(0);

	std::vector<float> a, b, c, c_cpu;

	if (!TA) {
		a = random_matrix(m,k);
	} else {
		a = random_matrix(k,m);
	}
	if (!TB) {
		b = random_matrix(k,n);
	} else {
		b = random_matrix(n,k);
	}

	int lda = (!TA)?k:m;
	int ldb = (!TB)?n:k;

	c = std::vector<float>(m*n, .0f);
	c_cpu = std::vector<float>(m*n, .0f);

	{
		Dummy dummy;
		auto start = std::chrono::high_resolution_clock::now();
		dummy.gemm(TA, TB, m, n, k, 1, a.data(), lda, b.data(), ldb, 1, c.data(), n);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end - start;
		std::cout << "OpenCL time = " << time.count() << " seconds" << std::endl;
	}

	{
		auto start = std::chrono::high_resolution_clock::now();
		gemm_cpu(TA,TB,m,n,k,1,a.data(),lda,b.data(),ldb,1,c_cpu.data(),n);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end - start;
		std::cout << "CPU time = " << time.count() << " seconds" << std::endl;
	}

	double sse = 0;
	for (int i = 0; i < m*n; ++i) {
		sse += pow(c_cpu[i]-c[i], 2);
	}

	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout.setf(std::ios::showpoint);

	std::cout << "SSE = " << sse << std::endl;
}

void main() {
	try {
		size_t M = 256;
		size_t N = 307200;
		size_t K = 27;

		for (int TA: {0, 1}) {
			for (int TB: {0, 1}) {
				test_opencl_accuracy(TA, TB, M, K, N);
			}
		}
	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
}

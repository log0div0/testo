
#include "Layer.hpp"
#include <stdexcept>

extern "C" {
#include "cuda.h"
}

namespace darknet {

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


void Layer::gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
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

#ifdef GPU

#include <math.h>

void Layer::gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A_gpu, int lda,
	float *B_gpu, int ldb,
	float BETA,
	float *C_gpu, int ldc)
{
	cublasHandle_t handle = blas_handle();
	cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
			(TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
	if (status != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("gemm_gpu");
	}
}

#endif

}

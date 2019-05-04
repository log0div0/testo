#include "gemm.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>

// void cblas_sgemm(
//     OPENBLAS_CONST enum CBLAS_ORDER Order,
//     OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
//     OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
//     OPENBLAS_CONST blasint M,
//     OPENBLAS_CONST blasint N,
//     OPENBLAS_CONST blasint K,
//     OPENBLAS_CONST float alpha,
//     OPENBLAS_CONST float *A,
//     OPENBLAS_CONST blasint lda,
//     OPENBLAS_CONST float *B,
//     OPENBLAS_CONST blasint ldb,
//     OPENBLAS_CONST float beta,
//     float *C,
//     OPENBLAS_CONST blasint ldc
// );

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    cblas_sgemm(
        CblasRowMajor,
        TA ? CblasTrans : CblasNoTrans,
        TB ? CblasTrans : CblasNoTrans,
        M,
        N,
        K,
        ALPHA,
        A,
        lda,
        B,
        ldb,
        BETA,
        C,
        ldc
    );
}

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A_gpu, int lda,
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#endif


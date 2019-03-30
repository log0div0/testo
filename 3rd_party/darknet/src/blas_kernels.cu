#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <assert.h>

extern "C" {
#include "blas.h"
#include "cuda.h"
#include "utils.h"
}

__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}

void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_scale_kernel(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
    }
}

void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    backward_scale_kernel<<<n, BLOCK>>>(x_norm, delta, batch, n, size, scale_updates);
    check_error(cudaPeekAtLastError());
}

__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    int num = n*size*batch;

    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_bias_conn_kernel(float *bias_updates, float *delta, int batch, int n)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int b;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        int i = b*n + index;
        sum += delta[i];
    }
    bias_updates[index] += sum;
}

__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}

void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
{
    if(size == 1){
        backward_bias_conn_kernel<<<cuda_gridsize(n), BLOCK>>>(bias_updates, delta, batch, n);
    }else{
        backward_bias_kernel<<<n, BLOCK>>>(bias_updates, delta, batch, n, size);
    }
    check_error(cudaPeekAtLastError());
}

__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;

    x[index] = (x[index] - mean[f])/(sqrtf(variance[f] + .00001f));
}

__global__ void normalize_delta_kernel(int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;

    delta[index] = delta[index] * 1.f/(sqrtf(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}

extern "C" void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    size_t N = batch*filters*spatial;
    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, mean_delta, variance_delta, batch, filters, spatial, delta);
    check_error(cudaPeekAtLastError());
}

__global__ void  variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    variance_delta[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance_delta[i] += delta[index]*(x[index] - mean[i]);
        }
    }
    variance_delta[i] *= -.5f * powf(variance[i] + .00001f, (float)(-3.f/2.f));
}

__global__ void accumulate_kernel(float *x, int n, int groups, float *sum)
{
    int k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    sum[i] = 0;
    for(k = 0; k < n; ++k){
        sum[i] += x[k*groups + i];
    }
}

__global__ void fast_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? delta[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (-1.f/sqrtf(variance[filter] + .00001f));
    }
}

__global__ void  fast_variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *= -.5f * powf(variance[filter] + .00001f, (float)(-3.f/2.f));
    }
}


__global__ void mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean_delta[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j*filters*spatial + i*spatial + k;
            mean_delta[i] += delta[index];
        }
    }
    mean_delta[i] *= (-1.f/sqrtf(variance[i] + .00001f));
}

extern "C" void mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    mean_delta_kernel<<<cuda_gridsize(filters), BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
    check_error(cudaPeekAtLastError());
}

extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    fast_mean_delta_kernel<<<filters, BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
    check_error(cudaPeekAtLastError());
}

extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    fast_variance_delta_kernel<<<filters, BLOCK>>>(x, delta, mean, variance, batch, filters, spatial, variance_delta);
    check_error(cudaPeekAtLastError());
}

__global__ void  mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1.f/(batch * spatial);
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}

__global__ void variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1.f/(batch * spatial - 1);
    int j,k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += powf((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}

__global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

__global__ void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

__global__ void const_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void add_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] += ALPHA;
}

__global__ void scal_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] *= ALPHA;
}

__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

__global__ void mul_kernel(int N, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] *= X[i*INCX];
}


extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

__global__ void  fast_mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? x[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}

__global__ void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? powf((x[index] - mean[filter]), 2) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}

extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    fast_mean_kernel<<<filters, BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}

extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    fast_variance_kernel<<<filters, BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}


extern "C" void mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    mean_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}

extern "C" void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    variance_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}

extern "C" void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

extern "C" void pow_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    pow_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void copy_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
}

extern "C" void mul_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
    mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void const_gpu(int N, float ALPHA, float * X, int INCX)
{
    const_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void add_gpu(int N, float ALPHA, float * X, int INCX)
{
    add_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void scal_gpu(int N, float ALPHA, float * X, int INCX)
{
    scal_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void fill_gpu(int N, float ALPHA, float * X, int INCX)
{
    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

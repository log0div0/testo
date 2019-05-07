#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "MaxPoolLayer.hpp"
#include "../Network.hpp"

extern "C" {
#include "cuda.h"
}

namespace darknet {

__global__ void forward_maxpool_layer_kernel(
	int n,
	int in_h, int in_w, int in_c,
	int stride_h, int stride_w,
	int size_h, int size_w,
	int pad_h, int pad_w,
	float *input, float *output,
	int *indexes)
{
	int h = (in_h + pad_h - size_h)/stride_h + 1;
	int w = (in_w + pad_w - size_w)/stride_w + 1;
	int c = in_c;

	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if(id >= n) return;

	int j = id % w;
	id /= w;
	int i = id % h;
	id /= h;
	int k = id % c;
	id /= c;
	int b = id;

	int w_offset = -pad_w/2;
	int h_offset = -pad_h/2;

	int out_index = j + w*(i + h*(k + c*b));
	float max = -INFINITY;
	int max_i = -1;
	for (int l = 0; l < size_h; ++l) {
		for (int m = 0; m < size_w; ++m) {
			int cur_h = h_offset + i*stride_h + l;
			int cur_w = w_offset + j*stride_w + m;
			int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
			int valid = (cur_h >= 0 && cur_h < in_h &&
					cur_w >= 0 && cur_w < in_w);
			float val = (valid != 0) ? input[index] : -INFINITY;
			max_i = (val > max) ? index : max_i;
			max   = (val > max) ? val   : max;
		}
	}
	output[out_index] = max;
	indexes[out_index] = max_i;
}

__global__ void backward_maxpool_layer_kernel(
	int n,
	int in_h, int in_w, int in_c,
	int stride_h, int stride_w,
	int size_h, int size_w,
	int pad_h, int pad_w,
	float *delta, float *prev_delta,
	int *indexes)
{
	int h = (in_h + pad_h - size_h)/stride_h + 1;
	int w = (in_w + pad_w - size_w)/stride_w + 1;
	int c = in_c;
	int area_h = (size_h-1)/stride_h;
	int area_w = (size_w-1)/stride_w;

	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if(id >= n) return;

	int index = id;
	int j = id % in_w;
	id /= in_w;
	int i = id % in_h;
	id /= in_h;
	int k = id % in_c;
	id /= in_c;
	int b = id;

	int w_offset = -pad_w/2;
	int h_offset = -pad_h/2;

	float d = 0;
	for (int l = -area_h; l < area_h+1; ++l) {
		for (int m = -area_w; m < area_w+1; ++m) {
			int out_w = (j-w_offset)/stride_w + m;
			int out_h = (i-h_offset)/stride_h + l;
			int out_index = out_w + w*(out_h + h*(k + c*b));
			int valid = (out_w >= 0 && out_w < w &&
					 out_h >= 0 && out_h < h);
			d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
		}
	}
	prev_delta[index] += d;
}

void MaxPoolLayer::forward_gpu(Network* net)
{
	size_t n = out_h*out_w*in_c*batch;

	forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(
		n,
		in_h, in_w, in_c,
		stride_h, stride_w,
		size_h, size_w,
		pad_h, pad_w,
		net->input_gpu, output_gpu,
		indexes_gpu);
	check_error(cudaPeekAtLastError());
}

void MaxPoolLayer::backward_gpu(Network* net)
{
	size_t n = in_h*in_w*in_c*batch;

	backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(
		n,
		in_h, in_w, in_c,
		stride_h, stride_w,
		size_h, size_w,
		pad_h, pad_w,
		delta_gpu, net->delta_gpu,
		indexes_gpu);
	check_error(cudaPeekAtLastError());
}


}

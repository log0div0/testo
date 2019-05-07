#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "ConvolutionalLayer.hpp"
#include "../Network.hpp"

extern "C" {
#include "cuda.h"
}

namespace darknet {

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
		const int height, const int width, const int ksize,
		const int pad,
		const int stride,
		const int height_col, const int width_col,
		float *data_col) {
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	for(; index < n; index += blockDim.x*gridDim.x){
		int w_out = index % width_col;
		int h_index = index / width_col;
		int h_out = h_index % height_col;
		int channel_in = h_index / height_col;
		int channel_out = channel_in * ksize * ksize;
		int h_in = h_out * stride - pad;
		int w_in = w_out * stride - pad;
		float* data_col_ptr = data_col;
		data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
		const float* data_im_ptr = data_im;
		data_im_ptr += (channel_in * height + h_in) * width + w_in;
		for (int i = 0; i < ksize; ++i) {
			for (int j = 0; j < ksize; ++j) {
				int h = h_in + i;
				int w = w_in + j;

				*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
					data_im_ptr[i * width + j] : 0;

				//*data_col_ptr = data_im_ptr[ii * width + jj];

				data_col_ptr += height_col * width_col;
			}
		}
	}
}

void ConvolutionalLayer::im2col_gpu(float *data_im, float *data_col) {
	// We are going to launch in_c * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (in_h + 2 * pad - size) / stride + 1;
	int width_col = (in_w + 2 * pad - size) / stride + 1;
	int num_kernels = in_c * height_col * width_col;
	im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(
		num_kernels, data_im, in_h, in_w, size, pad,
		stride, height_col,
		width_col, data_col);
}

__global__ void col2im_gpu_kernel(const int n, const float* data_col,
		const int height, const int width, const int ksize,
		const int pad,
		const int stride,
		const int height_col, const int width_col,
		float *data_im) {
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	for(; index < n; index += blockDim.x*gridDim.x){
		float val = 0;
		int w = index % width + pad;
		int h = (index / width) % height + pad;
		int c = index / (width * height);
		// compute the start and end of the output
		int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
		int w_col_end = min(w / stride + 1, width_col);
		int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
		int h_col_end = min(h / stride + 1, height_col);
		// equivalent implementation
		int offset =
			(c * ksize * ksize + h * ksize + w) * height_col * width_col;
		int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
		int coeff_w_col = (1 - stride * height_col * width_col);
		for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
			for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
				val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
			}
		}
		data_im[index] += val;
	}
}

void ConvolutionalLayer::col2im_gpu(float *data_col, float *data_im){
	// We are going to launch in_c * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (in_h + 2 * pad - size) / stride + 1;
	int width_col = (in_w + 2 * pad - size) / stride + 1;
	int num_kernels = in_c * in_h * in_w;
	col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(
		num_kernels, data_col, in_h, in_w, size, pad,
		stride, height_col,
		width_col, data_im);
}

}


#include "MaxPoolLayer.hpp"
#include "../Network.hpp"

extern "C" {
#include "cuda.h"
}

using namespace inipp;

namespace darknet {

MaxPoolLayer::MaxPoolLayer(const inisection& section,
	size_t batch,
	size_t w,
	size_t h,
	size_t c)
{
	this->batch = batch;
	in_h = h;
	in_w = w;
	in_c = c;

	size_h = section.get_int("size_h", section.get_int("size", 2));
    size_w = section.get_int("size_w", section.get_int("size", 2));
    stride_h = section.get_int("stride_h", section.get_int("stride", size_h));
    stride_w = section.get_int("stride_w", section.get_int("stride", size_w));
	pad_h = section.get_int("pad_h", section.get_int("pad", size_h / 2));
    pad_w = section.get_int("pad_w", section.get_int("pad", size_w / 2));

	if(!(h && w && c)) {
		throw std::runtime_error("Layer before maxpool layer must output image.");
	}

	out_w = (w + pad_w - size_w)/stride_w + 1;
	out_h = (h + pad_h - size_h)/stride_h + 1;
	out_c = c;
	outputs = out_h * out_w * out_c;
	int output_size = out_h * out_w * out_c * batch;
	indexes = (int*)calloc(output_size, sizeof(int));
	output =  (float*)calloc(output_size, sizeof(float));
	delta =   (float*)calloc(output_size, sizeof(float));
#ifdef GPU
	indexes_gpu = cuda_make_int_array(0, output_size);
	output_gpu  = cuda_make_array(output, output_size);
	delta_gpu   = cuda_make_array(delta, output_size);
#endif
	// fprintf(stderr, "max          %d x %d / %d x %d  %4zd x%4zd x%4zd   ->  %4d x%4d x%4d\n", size_w, size_h, stride_w, stride_h, w, h, c, out_w, out_h, out_c);
	workspace_size = 0;
}

MaxPoolLayer::~MaxPoolLayer() {
	free(indexes);
	free(output);
	free(delta);
#ifdef GPU
	cuda_free((float *)indexes_gpu);
	cuda_free(output_gpu);
	cuda_free(delta_gpu);
#endif
}

void MaxPoolLayer::forward(Network* net)
{
    int w_offset = -pad_w/2;
    int h_offset = -pad_h/2;

    int h = out_h;
    int w = out_w;

    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < in_c; ++k) {
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j) {
                    int out_index = j + w*(i + h*(k + in_c*b));
                    float max = -INFINITY;
                    int max_i = -1;
                    for (int n = 0; n < size_h; ++n) {
                        for (int m = 0; m < size_w; ++m) {
                            int cur_h = h_offset + i*stride_h + n;
                            int cur_w = w_offset + j*stride_w + m;
                            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
                            int valid = (cur_h >= 0 && cur_h < in_h &&
                                         cur_w >= 0 && cur_w < in_w);
                            float val = (valid != 0) ? net->input[index] : -INFINITY;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    output[out_index] = max;
                    indexes[out_index] = max_i;
                }
            }
        }
    }
}

void MaxPoolLayer::backward(Network* net)
{
    int h = out_h;
    int w = out_w;
    for (int i = 0; i < h*w*in_c*batch; ++i) {
        int index = indexes[i];
        net->delta[index] += delta[i];
    }
}

}

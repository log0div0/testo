
#include "MaxPoolLayer.hpp"
#include "../Network.hpp"

extern "C" {
#include "image.h"
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
	this->h = h;
	this->w = w;
	this->c = c;

	stride = section.get_int("stride", 1);
	size = section.get_int("size", stride);
	pad = section.get_int("padding", size-1);

	if(!(h && w && c)) {
		throw std::runtime_error("Layer before maxpool layer must output image.");
	}

	out_w = (w + pad - size)/stride + 1;
	out_h = (h + pad - size)/stride + 1;
	out_c = c;
	outputs = out_h * out_w * out_c;
	inputs = h*w*c;
	int output_size = out_h * out_w * out_c * batch;
	indexes = (int*)calloc(output_size, sizeof(int));
	output =  (float*)calloc(output_size, sizeof(float));
	delta =   (float*)calloc(output_size, sizeof(float));
#ifdef GPU
	indexes_gpu = cuda_make_int_array(0, output_size);
	output_gpu  = cuda_make_array(output, output_size);
	delta_gpu   = cuda_make_array(delta, output_size);
#endif
	fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, out_w, out_h, out_c);
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
    int w_offset = -pad/2;
    int h_offset = -pad/2;

    int h = out_h;
    int w = out_w;

    for(int b = 0; b < batch; ++b){
        for(int k = 0; k < c; ++k){
            for(int i = 0; i < h; ++i){
                for(int j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(int n = 0; n < size; ++n){
                        for(int m = 0; m < size; ++m){
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + this->w*(cur_h + this->h*(k + b*c));
                            int valid = (cur_h >= 0 && cur_h < this->h &&
                                         cur_w >= 0 && cur_w < this->w);
                            float val = (valid != 0) ? net->input[index] : -FLT_MAX;
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
    for (int i = 0; i < h*w*c*batch; ++i) {
        int index = indexes[i];
        net->delta[index] += delta[i];
    }
}

}

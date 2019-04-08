
#include "MaxPoolLayer.hpp"

extern "C" {
#include <maxpool_layer.h>
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
	forward = forward_maxpool_layer;
	backward = backward_maxpool_layer;
#ifdef GPU
	forward_gpu = forward_maxpool_layer_gpu;
	backward_gpu = backward_maxpool_layer_gpu;
	indexes_gpu = cuda_make_int_array(0, output_size);
	output_gpu  = cuda_make_array(output, output_size);
	delta_gpu   = cuda_make_array(delta, output_size);
#endif
	fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, out_w, out_h, out_c);
	workspace_size = 0;
}

}

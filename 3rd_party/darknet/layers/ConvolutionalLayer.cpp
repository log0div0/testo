
#include "ConvolutionalLayer.hpp"

extern "C" {
#include <convolutional_layer.h>
}

using namespace inipp;

namespace darknet {

size_t ConvolutionalLayer::get_workspace_size() const {
    return (size_t)out_h*out_w*size*size*c;
}

ConvolutionalLayer::ConvolutionalLayer(const inisection& section,
	size_t batch,
	size_t w,
	size_t h,
	size_t c)
{
    this->h = h;
    this->w = w;
    this->c = c;
    this->batch = batch;

	n = section.get_int("filters",1);
	size = section.get_int("size",1);
	stride = section.get_int("stride",1);
	pad = section.get_int("padding",0);
	batch_normalize = section.get_int("batch_normalize", 0);

	activation = get_activation((char*)section.get("activation", "logistic").c_str());

	if (!(h && w && c)) {
		throw std::runtime_error("Layer before convolutional layer must output image.");
	}








    weights = (float*)calloc(c*n*size*size, sizeof(float));
    weight_updates = (float*)calloc(c*n*size*size, sizeof(float));

    biases = (float*)calloc(n, sizeof(float));
    bias_updates = (float*)calloc(n, sizeof(float));

    nweights = c*n*size*size;
    nbiases = n;

    float scale = sqrt(2./(size*size*c));
    for (int i = 0; i < nweights; ++i) {
    	weights[i] = scale*rand_normal();
    }

    out_h = convolutional_out_height(*this);
    out_w = convolutional_out_width(*this);
    out_c = n;
    outputs = out_h * out_w * out_c;
    inputs = w * h * c;

    output = (float*)calloc(batch*outputs, sizeof(float));
    delta  = (float*)calloc(batch*outputs, sizeof(float));

    forward = forward_convolutional_layer;
    backward = backward_convolutional_layer;
    update = update_convolutional_layer;

    if (batch_normalize) {
        scales = (float*)calloc(n, sizeof(float));
        scale_updates = (float*)calloc(n, sizeof(float));
        for (int i = 0; i < n; ++i) {
            scales[i] = 1;
        }

        mean = (float*)calloc(n, sizeof(float));
        variance = (float*)calloc(n, sizeof(float));

        mean_delta = (float*)calloc(n, sizeof(float));
        variance_delta = (float*)calloc(n, sizeof(float));

        rolling_mean = (float*)calloc(n, sizeof(float));
        rolling_variance = (float*)calloc(n, sizeof(float));
        x = (float*)calloc(batch*outputs, sizeof(float));
        x_norm = (float*)calloc(batch*outputs, sizeof(float));
    }

#ifdef GPU
    forward_gpu = forward_convolutional_layer_gpu;
    backward_gpu = backward_convolutional_layer_gpu;
    update_gpu = update_convolutional_layer_gpu;

    if (use_gpu)
    {
        weights_gpu = cuda_make_array(weights, nweights);
        weight_updates_gpu = cuda_make_array(weight_updates, nweights);

        biases_gpu = cuda_make_array(biases, n);
        bias_updates_gpu = cuda_make_array(bias_updates, n);

        delta_gpu = cuda_make_array(delta, batch*out_h*out_w*n);
        output_gpu = cuda_make_array(output, batch*out_h*out_w*n);

        if (batch_normalize) {
            mean_gpu = cuda_make_array(mean, n);
            variance_gpu = cuda_make_array(variance, n);

            rolling_mean_gpu = cuda_make_array(mean, n);
            rolling_variance_gpu = cuda_make_array(variance, n);

            mean_delta_gpu = cuda_make_array(mean, n);
            variance_delta_gpu = cuda_make_array(variance, n);

            scales_gpu = cuda_make_array(scales, n);
            scale_updates_gpu = cuda_make_array(scale_updates, n);

            x_gpu = cuda_make_array(output, batch*out_h*out_w*n);
            x_norm_gpu = cuda_make_array(output, batch*out_h*out_w*n);
        }
    }
#endif
    workspace_size = get_workspace_size();

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, out_w, out_h, out_c, (2.0 * n * size*size*c * out_h*out_w)/1000000000.);
}

void ConvolutionalLayer::load_weights(FILE* fp)
{
    int num = c*n*size*size;
    fread(biases, sizeof(float), n, fp);
    if (batch_normalize){
        fread(scales, sizeof(float), n, fp);
        fread(rolling_mean, sizeof(float), n, fp);
        fread(rolling_variance, sizeof(float), n, fp);
    }
    fread(weights, sizeof(float), num, fp);
#ifdef GPU
    if(use_gpu){
        push_convolutional_layer(*this);
    }
#endif
}

void ConvolutionalLayer::save_weights(FILE* fp) const
{
#ifdef GPU
	if(use_gpu){
		pull_convolutional_layer(*this);
	}
#endif
	int num = nweights;
	fwrite(biases, sizeof(float), n, fp);
	if (batch_normalize){
		fwrite(scales, sizeof(float), n, fp);
		fwrite(rolling_mean, sizeof(float), n, fp);
		fwrite(rolling_variance, sizeof(float), n, fp);
	}
	fwrite(weights, sizeof(float), num, fp);
}

}

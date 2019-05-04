
#include "ConvolutionalLayer.hpp"
#include "../Network.hpp"

extern "C" {
#include <blas.h>
#include <gemm.h>
#include <activations.h>
}

using namespace inipp;

namespace darknet {

#define TWO_PI 6.2831853071795864769252866f

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
	static int haveSpare = 0;
	static double rand1, rand2;

	if(haveSpare)
	{
		haveSpare = 0;
		return sqrt(rand1) * sin(rand2);
	}

	haveSpare = 1;

	rand1 = rand() / ((double) RAND_MAX);
	if(rand1 < 1e-100) rand1 = 1e-100;
	rand1 = -2 * log(rand1);
	rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;

	return sqrt(rand1) * cos(rand2);
}

ConvolutionalLayer::ConvolutionalLayer(const inisection& section,
	size_t batch,
	size_t w,
	size_t h,
	size_t c)
{
	in_h = h;
	in_w = w;
	in_c = c;
	this->batch = batch;

	out_c = section.get_int("filters",1);
	size = section.get_int("size",1);
	stride = section.get_int("stride",1);
	pad = section.get_int("padding",0);
	batch_normalize = section.get_int("batch_normalize", 0);

	activation = get_activation((char*)section.get("activation", "logistic").c_str());

	if (!(h && w && c)) {
		throw std::runtime_error("Layer before convolutional layer must output image.");
	}

	nweights = in_c*out_c*size*size;
	nbiases = out_c;

	weights = (float*)calloc(nweights, sizeof(float));
	weight_updates = (float*)calloc(nweights, sizeof(float));

	biases = (float*)calloc(nbiases, sizeof(float));
	bias_updates = (float*)calloc(nbiases, sizeof(float));

	float scale = sqrt(2./(size*size*c));
	for (int i = 0; i < nweights; ++i) {
		weights[i] = scale*rand_normal();
	}

	out_h = get_out_height();
	out_w = get_out_width();
	outputs = out_h * out_w * out_c;

	output = (float*)calloc(batch*outputs, sizeof(float));
	delta  = (float*)calloc(batch*outputs, sizeof(float));

	if (batch_normalize) {
		scales = (float*)calloc(out_c, sizeof(float));
		scale_updates = (float*)calloc(out_c, sizeof(float));
		for (int i = 0; i < out_c; ++i) {
			scales[i] = 1;
		}

		mean = (float*)calloc(out_c, sizeof(float));
		variance = (float*)calloc(out_c, sizeof(float));

		mean_delta = (float*)calloc(out_c, sizeof(float));
		variance_delta = (float*)calloc(out_c, sizeof(float));

		rolling_mean = (float*)calloc(out_c, sizeof(float));
		rolling_variance = (float*)calloc(out_c, sizeof(float));
		x = (float*)calloc(batch*outputs, sizeof(float));
		x_norm = (float*)calloc(batch*outputs, sizeof(float));
	}

#ifdef GPU
	if (use_gpu)
	{
		weights_gpu = cuda_make_array(weights, nweights);
		weight_updates_gpu = cuda_make_array(weight_updates, nweights);

		biases_gpu = cuda_make_array(biases, nbiases);
		bias_updates_gpu = cuda_make_array(bias_updates, nbiases);

		delta_gpu = cuda_make_array(delta, batch*out_h*out_w*out_c);
		output_gpu = cuda_make_array(output, batch*out_h*out_w*out_c);

		if (batch_normalize) {
			mean_gpu = cuda_make_array(mean, out_c);
			variance_gpu = cuda_make_array(variance, out_c);

			rolling_mean_gpu = cuda_make_array(mean, out_c);
			rolling_variance_gpu = cuda_make_array(variance, out_c);

			mean_delta_gpu = cuda_make_array(mean, out_c);
			variance_delta_gpu = cuda_make_array(variance, out_c);

			scales_gpu = cuda_make_array(scales, out_c);
			scale_updates_gpu = cuda_make_array(scale_updates, out_c);

			x_gpu = cuda_make_array(output, batch*out_h*out_w*out_c);
			x_norm_gpu = cuda_make_array(output, batch*out_h*out_w*out_c);
		}
	}
#endif
	workspace_size = get_workspace_size();

	fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", out_c, size, size, stride, in_w, in_h, in_c, out_w, out_h, out_c, (2.0 * out_c * size*size*c * out_h*out_w)/1000000000.);
}

ConvolutionalLayer::~ConvolutionalLayer() {
	free(weights);
	free(weight_updates);
	free(biases);
	free(bias_updates);
	free(output);
	free(delta);
	if (batch_normalize) {
		free(scales);
		free(scale_updates);
		free(mean);
		free(variance);
		free(mean_delta);
		free(variance_delta);
		free(rolling_mean);
		free(rolling_variance);
		free(x);
		free(x_norm);
	}
#ifdef GPU
	if (use_gpu) {
		cuda_free(weights_gpu);
		cuda_free(weight_updates_gpu);
		cuda_free(biases_gpu);
		cuda_free(bias_updates_gpu);
		cuda_free(output_gpu);
		cuda_free(delta_gpu);
		if (batch_normalize) {
			cuda_free(mean_gpu);
			cuda_free(variance_gpu);
			cuda_free(rolling_mean_gpu);
			cuda_free(rolling_variance_gpu);
			cuda_free(mean_delta_gpu);
			cuda_free(variance_delta_gpu);
			cuda_free(scales_gpu);
			cuda_free(scale_updates_gpu);
			cuda_free(x_gpu);
			cuda_free(x_norm_gpu);
		}
	}
#endif
}

void ConvolutionalLayer::load_weights(std::istream& stream)
{
	stream.read((char*)biases, sizeof(float) * out_c);
	if (batch_normalize) {
		stream.read((char*)scales, sizeof(float) * out_c);
		stream.read((char*)rolling_mean, sizeof(float) * out_c);
		stream.read((char*)rolling_variance, sizeof(float) * out_c);
	}
	stream.read((char*)weights, sizeof(float) * nweights);
#ifdef GPU
	if(use_gpu){
		push();
	}
#endif
}

void ConvolutionalLayer::save_weights(FILE* fp) const
{
#ifdef GPU
	if(use_gpu){
		pull();
	}
#endif
	fwrite(biases, sizeof(float), out_c, fp);
	if (batch_normalize){
		fwrite(scales, sizeof(float), out_c, fp);
		fwrite(rolling_mean, sizeof(float), out_c, fp);
		fwrite(rolling_variance, sizeof(float), out_c, fp);
	}
	fwrite(weights, sizeof(float), nweights, fp);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
	int i,j,b;
	for(b = 0; b < batch; ++b){
		for(i = 0; i < n; ++i){
			for(j = 0; j < size; ++j){
				output[(b*n + i)*size + j] += biases[i];
			}
		}
	}
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
	int i,j,b;
	for(b = 0; b < batch; ++b){
		for(i = 0; i < n; ++i){
			for(j = 0; j < size; ++j){
				output[(b*n + i)*size + j] *= scales[i];
			}
		}
	}
}

float sum_array(float *a, int n)
{
	int i;
	float sum = 0;
	for(i = 0; i < n; ++i) sum += a[i];
	return sum;
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
	int i,b;
	for(b = 0; b < batch; ++b){
		for(i = 0; i < n; ++i){
			bias_updates[i] += sum_array(delta+size*(i+b*n), size);
		}
	}
}

void ConvolutionalLayer::forward(Network* net)
{
	fill_cpu(outputs*batch, 0, output, 1);

	int m = out_c;
	int k = size*size*in_c;
	int n = out_w*out_h;
	for (int i = 0; i < batch; ++i) {
		float *a = weights;
		float *b = net->workspace;
		float *c = output + i*n*m;
		float *im =  net->input + i*in_c*in_h*in_w;

		if (size == 1) {
			b = im;
		} else {
			im2col_cpu(im, b);
		}
		gemm_cpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
	}

	if (batch_normalize) {
		copy_cpu(outputs*batch, output, 1, x, 1);
		if (net->train) {
			mean_cpu(output, batch, out_c, out_h*out_w, mean);
			variance_cpu(output, mean, batch, out_c, out_h*out_w, variance);

			scal_cpu(out_c, .99, rolling_mean, 1);
			axpy_cpu(out_c, .01, mean, 1, rolling_mean, 1);
			scal_cpu(out_c, .99, rolling_variance, 1);
			axpy_cpu(out_c, .01, variance, 1, rolling_variance, 1);

			normalize_cpu(output, mean, variance, batch, out_c, out_h*out_w);
			copy_cpu(outputs*batch, output, 1, x_norm, 1);
		} else {
			normalize_cpu(output, rolling_mean, rolling_variance, batch, out_c, out_h*out_w);
		}
		scale_bias(output, scales, batch, out_c, out_h*out_w);
		add_bias(output, biases, batch, out_c, out_h*out_w);
	} else {
		add_bias(output, biases, batch, out_c, out_h*out_w);
	}

	activate_array(output, outputs*batch, activation);
}

#ifdef GPU

void ConvolutionalLayer::forward_gpu(Network* net)
{
	fill_gpu(outputs*batch, 0, output_gpu, 1);

	int m = out_c;
	int k = size*size*in_c;
	int n = out_w*out_h;
	for (int i = 0; i < batch; ++i) {
		float *a = weights_gpu;
		float *b = net->workspace;
		float *c = output_gpu + i*n*m;
		float *im = net->input_gpu + i*in_c*in_h*in_w;

		if (size == 1){
			b = im;
		} else {
			im2col_gpu(im, b);
		}
		gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
	}

	if (batch_normalize) {
		copy_gpu(outputs*batch, output_gpu, 1, x_gpu, 1);
		if (net->train) {
			fast_mean_gpu(output_gpu, batch, out_c, out_h*out_w, mean_gpu);
			fast_variance_gpu(output_gpu, mean_gpu, batch, out_c, out_h*out_w, variance_gpu);

			scal_gpu(out_c, .99, rolling_mean_gpu, 1);
			axpy_gpu(out_c, .01, mean_gpu, 1, rolling_mean_gpu, 1);
			scal_gpu(out_c, .99, rolling_variance_gpu, 1);
			axpy_gpu(out_c, .01, variance_gpu, 1, rolling_variance_gpu, 1);

			copy_gpu(outputs*batch, output_gpu, 1, x_gpu, 1);
			normalize_gpu(output_gpu, mean_gpu, variance_gpu, batch, out_c, out_h*out_w);
			copy_gpu(outputs*batch, output_gpu, 1, x_norm_gpu, 1);

			scale_bias_gpu(output_gpu, scales_gpu, batch, out_c, out_h*out_w);
			add_bias_gpu(output_gpu, biases_gpu, batch, out_c, out_w*out_h);
		} else {
			normalize_gpu(output_gpu, rolling_mean_gpu, rolling_variance_gpu, batch, out_c, out_h*out_w);
			scale_bias_gpu(output_gpu, scales_gpu, batch, out_c, out_h*out_w);
			add_bias_gpu(output_gpu, biases_gpu, batch, out_c, out_w*out_h);
		}
	} else {
		add_bias_gpu(output_gpu, biases_gpu, batch, out_c, out_w*out_h);
	}

	activate_array_gpu(output_gpu, outputs*batch, activation);
}

#endif

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
	int i,b,f;
	for(f = 0; f < n; ++f){
		float sum = 0;
		for(b = 0; b < batch; ++b){
			for(i = 0; i < size; ++i){
				int index = i + size*(f + n*b);
				sum += delta[index] * x_norm[index];
			}
		}
		scale_updates[f] += sum;
	}
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

	int i,j,k;
	for(i = 0; i < filters; ++i){
		mean_delta[i] = 0;
		for (j = 0; j < batch; ++j) {
			for (k = 0; k < spatial; ++k) {
				int index = j*filters*spatial + i*spatial + k;
				mean_delta[i] += delta[index];
			}
		}
		mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
	}
}

void variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

	int i,j,k;
	for(i = 0; i < filters; ++i){
		variance_delta[i] = 0;
		for(j = 0; j < batch; ++j){
			for(k = 0; k < spatial; ++k){
				int index = j*filters*spatial + i*spatial + k;
				variance_delta[i] += delta[index]*(x[index] - mean[i]);
			}
		}
		variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
	}
}

void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
	int f, j, k;
	for(j = 0; j < batch; ++j){
		for(f = 0; f < filters; ++f){
			for(k = 0; k < spatial; ++k){
				int index = j*filters*spatial + f*spatial + k;
				delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
			}
		}
	}
}

void ConvolutionalLayer::backward(Network* net)
{
	int m = out_c;
	int n = size*size*in_c;
	int k = out_w*out_h;

	gradient_array(output, outputs*batch, activation, delta);

	if (batch_normalize) {
		if (!net->train) {
			mean = rolling_mean;
			variance = rolling_variance;
		}
		backward_bias(bias_updates, delta, batch, out_c, out_w*out_h);
		backward_scale_cpu(x_norm, delta, batch, out_c, out_w*out_h, scale_updates);

		scale_bias(delta, scales, batch, out_c, out_h*out_w);

		mean_delta_cpu(delta, variance, batch, out_c, out_w*out_h, mean_delta);
		variance_delta_cpu(x, delta, mean, variance, batch, out_c, out_w*out_h, variance_delta);
		normalize_delta_cpu(x, mean, variance, mean_delta, variance_delta, batch, out_c, out_w*out_h, delta);
	} else {
		backward_bias(bias_updates, delta, batch, out_c, k);
	}

	for (int i = 0; i < batch; ++i) {
		float *a = delta + i*m*k;
		float *b = net->workspace;
		float *c = weight_updates;

		float *im  = net->input + i*in_c*in_h*in_w;
		float *imd = net->delta + i*in_c*in_h*in_w;

		if(size == 1){
			b = im;
		} else {
			im2col_cpu(im, b);
		}

		gemm_cpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

		if (net->delta) {
			a = weights;
			b = delta + i*m*k;
			c = net->workspace;
			if (size == 1) {
				c = imd;
			}

			gemm_cpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

			if (size != 1) {
				col2im_cpu(net->workspace, imd);
			}
		}
	}
}

#ifdef GPU

void ConvolutionalLayer::backward_gpu(Network* net)
{
	gradient_array_gpu(output_gpu, outputs*batch, activation, delta_gpu);

	if (batch_normalize) {
		if (!net->train) {
			mean_gpu = rolling_mean_gpu;
			variance_gpu = rolling_variance_gpu;
		}
		backward_bias_gpu(bias_updates_gpu, delta_gpu, batch, out_c, out_w*out_h);
		backward_scale_gpu(x_norm_gpu, delta_gpu, batch, out_c, out_w*out_h, scale_updates_gpu);

		scale_bias_gpu(delta_gpu, scales_gpu, batch, out_c, out_h*out_w);

		fast_mean_delta_gpu(delta_gpu, variance_gpu, batch, out_c, out_w*out_h, mean_delta_gpu);
		fast_variance_delta_gpu(x_gpu, delta_gpu, mean_gpu, variance_gpu, batch, out_c, out_w*out_h, variance_delta_gpu);
		normalize_delta_gpu(x_gpu, mean_gpu, variance_gpu, mean_delta_gpu, variance_delta_gpu, batch, out_c, out_w*out_h, delta_gpu);
	} else {
		backward_bias_gpu(bias_updates_gpu, delta_gpu, batch, out_c, out_w*out_h);
	}
	//float *original_input = net->input_gpu;

	int m = out_c;
	int n = size*size*in_c;
	int k = out_w*out_h;

	int i;
	for(i = 0; i < batch; ++i){
		float *a = delta_gpu + i*m*k;
		float *b = net->workspace;
		float *c = weight_updates_gpu;

		float *im  = net->input_gpu+i*in_c*in_h*in_w;
		float *imd = net->delta_gpu+i*in_c*in_h*in_w;

		im2col_gpu(im, b);
		gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

		if (net->delta_gpu) {
			a = weights_gpu;
			b = delta_gpu + i*m*k;
			c = net->workspace;
			if (size == 1) {
				c = imd;
			}

			gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

			if (size != 1) {
				col2im_gpu(net->workspace, imd);
			}
		}
	}
}

#endif

void ConvolutionalLayer::update(Network* net, float learning_rate, float momentum, float decay)
{
	axpy_cpu(out_c, learning_rate/batch, bias_updates, 1, biases, 1);
	scal_cpu(out_c, momentum, bias_updates, 1);

	if (scales) {
		axpy_cpu(out_c, learning_rate/batch, scale_updates, 1, scales, 1);
		scal_cpu(out_c, momentum, scale_updates, 1);
	}

	axpy_cpu(nweights, -decay*batch, weights, 1, weight_updates, 1);
	axpy_cpu(nweights, learning_rate/batch, weight_updates, 1, weights, 1);
	scal_cpu(nweights, momentum, weight_updates, 1);
}

#ifdef GPU

void ConvolutionalLayer::update_gpu(Network* net, float learning_rate, float momentum, float decay)
{
	axpy_gpu(nweights, -decay*batch, weights_gpu, 1, weight_updates_gpu, 1);
	axpy_gpu(nweights, learning_rate/batch, weight_updates_gpu, 1, weights_gpu, 1);
	scal_gpu(nweights, momentum, weight_updates_gpu, 1);

	axpy_gpu(nbiases, learning_rate/batch, bias_updates_gpu, 1, biases_gpu, 1);
	scal_gpu(nbiases, momentum, bias_updates_gpu, 1);

	if (scales_gpu) {
		axpy_gpu(out_c, learning_rate/batch, scale_updates_gpu, 1, scales_gpu, 1);
		scal_gpu(out_c, momentum, scale_updates_gpu, 1);
	}
}

void ConvolutionalLayer::pull() const
{
	cuda_pull_array(weights_gpu, weights, nweights);
	cuda_pull_array(biases_gpu, biases, out_c);
	cuda_pull_array(weight_updates_gpu, weight_updates, nweights);
	cuda_pull_array(bias_updates_gpu, bias_updates, out_c);
	if (batch_normalize) {
		cuda_pull_array(scales_gpu, scales, out_c);
		cuda_pull_array(rolling_mean_gpu, rolling_mean, out_c);
		cuda_pull_array(rolling_variance_gpu, rolling_variance, out_c);
	}
}

void ConvolutionalLayer::push() const
{
	cuda_push_array(weights_gpu, weights, nweights);
	cuda_push_array(biases_gpu, biases, out_c);
	cuda_push_array(weight_updates_gpu, weight_updates, nweights);
	cuda_push_array(bias_updates_gpu, bias_updates, out_c);
	if (batch_normalize) {
		cuda_push_array(scales_gpu, scales, out_c);
		cuda_push_array(rolling_mean_gpu, rolling_mean, out_c);
		cuda_push_array(rolling_variance_gpu, rolling_variance, out_c);
	}
}

#endif

float im2col_get_pixel(float *im, int height, int width, int channels,
						int row, int col, int channel, int pad)
{
	row -= pad;
	col -= pad;

	if (row < 0 || col < 0 ||
		row >= height || col >= width) return 0;
	return im[col + width*(row + height*channel)];
}

void ConvolutionalLayer::im2col_cpu(float* data_im, float* data_col)
{
	int height_col = (in_h + 2*pad - size) / stride + 1;
	int width_col = (in_w + 2*pad - size) / stride + 1;

	int channels_col = in_c * size * size;
	for (int c = 0; c < channels_col; ++c) {
		int w_offset = c % size;
		int h_offset = (c / size) % size;
		int c_im = c / size / size;
		for (int h = 0; h < height_col; ++h) {
			for (int w = 0; w < width_col; ++w) {
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				data_col[col_index] = im2col_get_pixel(data_im, in_h, in_w, in_c,
						im_row, im_col, c_im, pad);
			}
		}
	}
}

void col2im_add_pixel(float *im, int height, int width, int channels,
						int row, int col, int channel, int pad, float val)
{
	row -= pad;
	col -= pad;

	if (row < 0 || col < 0 ||
		row >= height || col >= width) return;
	im[col + width*(row + height*channel)] += val;
}

void ConvolutionalLayer::col2im_cpu(float* data_col, float* data_im)
{
	int height_col = (in_h + 2*pad - size) / stride + 1;
	int width_col = (in_w + 2*pad - size) / stride + 1;

	int channels_col = in_c * size * size;
	for (int c = 0; c < channels_col; ++c) {
		int w_offset = c % size;
		int h_offset = (c / size) % size;
		int c_im = c / size / size;
		for (int h = 0; h < height_col; ++h) {
			for (int w = 0; w < width_col; ++w) {
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				double val = data_col[col_index];
				col2im_add_pixel(data_im, in_h, in_w, in_c,
						im_row, im_col, c_im, pad, val);
			}
		}
	}
}

size_t ConvolutionalLayer::get_workspace_size() const {
	return (size_t)out_h*out_w*size*size*in_c;
}

int ConvolutionalLayer::get_out_height() const
{
	return (in_h + 2*pad - size) / stride + 1;
}

int ConvolutionalLayer::get_out_width() const
{
	return (in_w + 2*pad - size) / stride + 1;
}

}

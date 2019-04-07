
#include "Network.hpp"
#include <stdexcept>
#include <inipp.hh>
#include <assert.h>

#include "layers/ConvolutionalLayer.hpp"
#include "layers/MaxPoolLayer.hpp"

extern "C" {

#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"

void save_convolutional_weights(layer l, FILE *fp);
void save_batchnorm_weights(layer l, FILE *fp);
void load_convolutional_weights(layer l, FILE *fp);
void load_batchnorm_weights(layer l, FILE *fp);
}

using namespace inipp;

namespace darknet {

struct size_params {
	int batch;
	int inputs;
	int h;
	int w;
	int c;
	network *net;
};

Network::Network(const std::string& path): network({})
{
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}
	inifile ini(file);

	n = ini.sections().size();
	layers = (layer*)calloc(n, sizeof(layer));
	batch = ini.get_int("batch", 1);
	learning_rate = ini.get_float("learning_rate", .001);
	momentum = ini.get_float("momentum", .9);
	decay = ini.get_float("decay", .0001);
	h = ini.get_int("height", 0);
	w = ini.get_int("width", 0);
	c = ini.get_int("channels", 0);
	inputs = h * w * c;

	if (!inputs && !(h && w && c)) {
		throw std::runtime_error("No input parameters supplied");
	}

	size_params params;
	params.h = h;
	params.w = w;
	params.c = c;
	params.inputs = inputs;
	params.batch = batch;
	params.net = this;

	size_t workspace_size = 0;
	int count = 0;
	fprintf(stderr, "layer     filters    size              input                output\n");
	for (auto& section: ini.sections()) {
		fprintf(stderr, "%5d ", count);

		layer l = {};
		if (section.name() == "convolutional") {
			l = ConvolutionalLayer(section, params.batch, params.w, params.h, params.c);
		} else if (section.name() == "maxpool"){
			l = MaxPoolLayer(section, params.batch, params.w, params.h, params.c);
		} else {
			throw std::runtime_error("Unknown layer type: " + section.name());
		}
		layers[count] = l;

		params.h = l.out_h;
		params.w = l.out_w;
		params.c = l.out_c;
		params.inputs = l.outputs;
		if (l.workspace_size > workspace_size) {
			workspace_size = l.workspace_size;
		}
		++count;
	}
	layer out = back();
	outputs = out.outputs;
	input = (float*)calloc(inputs*batch, sizeof(float));
#ifdef GPU
	output_gpu = out.output_gpu;
	input_gpu = cuda_make_array(input, inputs*batch);
#endif
	if (workspace_size) {
#ifdef GPU
		if (use_gpu) {
			workspace = (float*)cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
		}
		else
#endif
		{
			workspace = (float*)calloc(1, workspace_size);
		}
	}
}

Network::~Network() {
	if (layers) {
		for (int i = 0; i < n; ++i) {
			free_layer(layers[i]);
		}
		free(layers);
	}
	if(input) free(input);
#ifdef GPU
	if(input_gpu) cuda_free(input_gpu);
#endif
}

Network::Network(Network&& other): network(other) {
	(network&)other = {};
}

void Network::load_weights(const std::string& weights_file_path) {
	FILE *fp = fopen(weights_file_path.c_str(), "rb");
	if(!fp) {
		throw std::runtime_error("Failed to open file " + weights_file_path);
	}

	for(int i = 0; i < n; ++i){
		layer l = layers[i];
		if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
			load_convolutional_weights(l, fp);
		}
		if(l.type == BATCHNORM){
			load_batchnorm_weights(l, fp);
		}
	}
	fclose(fp);
}

void Network::save_weights(const std::string& weights_file_path) {
	FILE *fp = fopen(weights_file_path.c_str(), "wb");
	if(!fp) {
		throw std::runtime_error("Failed to open file " + weights_file_path);
	}

	for(int i = 0; i < n; ++i){
		layer l = layers[i];
		if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
			save_convolutional_weights(l, fp);
		} if(l.type == BATCHNORM){
			save_batchnorm_weights(l, fp);
		}
	}
	fclose(fp);
}

void Network::forward() {
	network backup = *this;
#ifdef GPU
	if (use_gpu) {
		cuda_push_array(input_gpu, input, inputs*batch);

		for(int i = 0; i < n; ++i){
			layer l = layers[i];
			if(l.delta){
				fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
			}
			if(l.delta_gpu){
				fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
			}
			l.forward_gpu(l, *this);
			input_gpu = l.output_gpu;
			input = l.output;
		}
		layer l = layers[n - 1];
		cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
	}
	else
#endif
	{
		for(int i = 0; i < n; ++i){
			layer l = layers[i];
			if(l.delta){
				fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
			}
			l.forward(l, *this);
			input = l.output;
		}
	}
	*(network*)this = backup;
}

void Network::backward() {
	network backup = *this;
#ifdef GPU
	if (use_gpu) {
		layer l = layers[n - 1];
		cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
		for(int i = n-1; i >= 0; --i){
			layer l = layers[i];
			if(i == 0){
				*(network*)this = backup;
			}else{
				layer prev = layers[i-1];
				input = prev.output;
				delta = prev.delta;
				input_gpu = prev.output_gpu;
				delta_gpu = prev.delta_gpu;
			}
			l.backward_gpu(l, *this);
		}
	}
	else
#endif
	{
		for(int i = n-1; i >= 0; --i){
			layer l = layers[i];
			if(i == 0){
				*(network*)this = backup;
			}else{
				layer prev = layers[i-1];
				input = prev.output;
				delta = prev.delta;
			}
			l.backward(l, *this);
		}
	}
	*(network*)this = backup;
}

void Network::update() {
#ifdef GPU
	if (use_gpu) {
		for(int i = 0; i < n; ++i){
			layer l = layers[i];
			if(l.update_gpu){
				l.update_gpu(l, *this);
			}
		}
	}
	else
#endif
	{
		for(int i = 0; i < n; ++i){
			layer l = layers[i];
			if(l.update){
				l.update(l, *this);
			}
		}
	}
}

}

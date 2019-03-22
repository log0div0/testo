
#include "Network.hpp"
#include <stdexcept>
#include <inipp.hh>
#include <assert.h>

#include "layers/ConvolutionalLayer.hpp"
#include "layers/YoloLayer.hpp"
#include "layers/MaxPoolLayer.hpp"

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

Network::Network(const std::string& path
#ifdef GPU
	, int gpu
#endif
	): network({})
{
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}
	inifile ini(file);

	n = ini.sections().size();
	layers = (layer*)calloc(n, sizeof(layer));
#ifdef GPU
	gpu_index = gpu;
	if (gpu_index >= 0) {
		cuda_set_device(gpu_index);
	}
#endif
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
		} else if (section.name() == "yolo") {
			l = YoloLayer(section, params.batch, params.w, params.h, params.c);
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
	truths = out.outputs;
	if (out.truths) {
		truths = out.truths;
	}
	input = (float*)calloc(inputs*batch, sizeof(float));
	truth = (float*)calloc(truths*batch, sizeof(float));
#ifdef GPU
	output_gpu = out.output_gpu;
	input_gpu = cuda_make_array(input, inputs*batch);
	truth_gpu = cuda_make_array(truth, truths*batch);
#endif
	if(workspace_size){
#ifdef GPU
		if(gpu_index >= 0){
			workspace = (float*)cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
		}else {
			workspace = (float*)calloc(1, workspace_size);
		}
#else
		workspace = (float*)calloc(1, workspace_size);
#endif
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
	if(truth) free(truth);
#ifdef GPU
	if(input_gpu) cuda_free(input_gpu);
	if(truth_gpu) cuda_free(truth_gpu);
#endif
}

Network::Network(Network&& other): network(other) {
	(network&)other = {};
}

void Network::load_weights(const std::string& weights_file_path) {
	::load_weights(this, (char*)weights_file_path.c_str());
}

void Network::save_weights(const std::string& weights_file_path) {
	::save_weights(this, (char*)weights_file_path.c_str());
}

void Network::forward() {
	network backup = *this;
#ifdef GPU
	if(gpu_index >= 0){
		forward_network_gpu(this);
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

}

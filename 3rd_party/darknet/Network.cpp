
#include "Network.hpp"
#include <stdexcept>
#include <assert.h>
#include <inipp.hh>

#include "layers/ConvolutionalLayer.hpp"
#include "layers/MaxPoolLayer.hpp"

using namespace inipp;

namespace darknet {

struct size_params {
	int batch;
	int inputs;
	int h;
	int w;
	int c;
};

Network::Network(const std::string& path,
	int batch_, int w_, int h_, int c_)
{
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}
	new (this) Network(file, batch_, w_, h_, c_);
}

Network::Network(std::istream& stream,
	int batch_, int w_, int h_, int c_)
{
	inifile ini(stream);

	batch = batch_;
	h = h_;
	w = w_;
	c = c_;
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

	size_t workspace_size = 0;
	int count = 0;
	fprintf(stderr, "layer     filters    size              input                output\n");
	for (auto& section: ini.sections()) {
		fprintf(stderr, "%5d ", count);

		std::unique_ptr<Layer> l;
		if (section.name() == "convolutional") {
			l.reset(new ConvolutionalLayer(section, params.batch, params.w, params.h, params.c));
		} else if (section.name() == "maxpool"){
			l.reset(new MaxPoolLayer(section, params.batch, params.w, params.h, params.c));
		} else {
			throw std::runtime_error("Unknown layer type: " + section.name());
		}

		params.h = l->out_h;
		params.w = l->out_w;
		params.c = l->out_c;
		params.inputs = l->outputs;
		if (l->workspace_size > workspace_size) {
			workspace_size = l->workspace_size;
		}
		layers.push_back(std::move(l));
		++count;
	}
	auto& out = layers.back();
	outputs = out->outputs;
	input = (float*)calloc(inputs*batch, sizeof(float));
#ifdef GPU
	input_gpu = cuda_make_array(input, inputs*batch);
#endif
	if (workspace_size) {
#ifdef GPU
		if (use_gpu) {
			workspace = cuda_make_array(nullptr, workspace_size);
		}
		else
#endif
		{
			workspace = (float*)calloc(workspace_size, sizeof(float));
		}
	}
}

Network::~Network() {
	if(input) free(input);
#ifdef GPU
	if(input_gpu) cuda_free(input_gpu);
#endif
}

void Network::load_weights(const std::string& weights_file_path) {
	std::ifstream file(weights_file_path, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + weights_file_path);
	}
	load_weights(file);
}

void Network::load_weights(std::istream& weights_file) {
	for (auto& l: layers) {
		l->load_weights(weights_file);
	}
}

void Network::save_weights(const std::string& weights_file_path) {
	FILE *fp = fopen(weights_file_path.c_str(), "wb");
	if(!fp) {
		throw std::runtime_error("Failed to open file " + weights_file_path);
	}

	for (auto& l: layers) {
		l->save_weights(fp);
	}
	fclose(fp);
}

void Network::forward() {
	float* input_backup = input;
#ifdef GPU
	float* input_gpu_backup = input_gpu;
	if (use_gpu) {
		cuda_push_array(input_gpu, input, inputs*batch);

		for (int i = 0; i < layers.size(); ++i) {
			auto& l = layers[i];
			if (l->delta) {
				fill_cpu(l->outputs * l->batch, 0, l->delta, 1);
			}
			if (l->delta_gpu) {
				fill_gpu(l->outputs * l->batch, 0, l->delta_gpu, 1);
			}
			l->forward_gpu(this);
			input_gpu = l->output_gpu;
			input = l->output;
		}
		auto& l = layers.back();
		cuda_pull_array(l->output_gpu, l->output, l->outputs*l->batch);
	}
	else
#endif
	{
		for (size_t i = 0; i < layers.size(); ++i) {
			auto& l = layers[i];
			if (l->delta) {
				fill_cpu(l->outputs * l->batch, 0, l->delta, 1);
			}
			l->forward(this);
			input = l->output;
		}
	}
	input = input_backup;
#ifdef GPU
	input_gpu = input_gpu_backup;
#endif
}

void Network::backward() {
	float* input_backup = input;
	float* delta_backup = delta;
#ifdef GPU
	float* input_gpu_backup = input_gpu;
	float* delta_gpu_backup = delta_gpu;
	if (use_gpu) {
		auto& l = layers.back();
		cuda_push_array(l->delta_gpu, l->delta, l->batch*l->outputs);
		for (int i = layers.size()-1; i >= 0; --i) {
			auto& l = layers[i];
			if (i == 0) {
				input = input_backup;
				input_gpu = input_gpu_backup;
				delta = delta_backup;
				delta_gpu = delta_gpu_backup;
			} else {
				auto& prev = layers[i-1];
				input = prev->output;
				delta = prev->delta;
				input_gpu = prev->output_gpu;
				delta_gpu = prev->delta_gpu;
			}
			l->backward_gpu(this);
		}
	}
	else
#endif
	{
		for (int i = layers.size()-1; i >= 0; --i) {
			auto& l = layers[i];
			if (i == 0) {
				input = input_backup;
				delta = delta_backup;
			} else {
				auto& prev = layers[i-1];
				input = prev->output;
				delta = prev->delta;
			}
			l->backward(this);
		}
	}
	input = input_backup;
	delta = delta_backup;
#ifdef GPU
	input_gpu = input_gpu_backup;
	delta_gpu = delta_gpu_backup;
#endif
}

void Network::update(float learning_rate, float momentum, float decay) {
#ifdef GPU
	if (use_gpu) {
		for (auto& l: layers) {
			l->update_gpu(this, learning_rate, momentum, decay);
		}
	}
	else
#endif
	{
		for (auto& l: layers) {
			l->update(this, learning_rate, momentum, decay);
		}
	}
}

}

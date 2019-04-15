
#pragma once

#include "include/darknet.h"
#include <istream>

namespace darknet {

struct Network;

struct Layer {
	Layer() {};
	virtual ~Layer() {};

	Layer(const Layer& other) = delete;
	Layer& operator=(const Layer& other) = delete;

	virtual void load_weights(std::istream& stream) = 0;
	virtual void save_weights(FILE* fp) const = 0;

	virtual void forward(Network* network) = 0;
	virtual void backward(Network* network) = 0;
	virtual void update(Network* network, float learning_rate, float momentum, float decay) = 0;
#ifdef GPU
	virtual void forward_gpu(Network* network) = 0;
	virtual void backward_gpu(Network* network) = 0;
	virtual void update_gpu(Network* network, float learning_rate, float momentum, float decay) = 0;
#endif

	int out_w = 0;
	int out_h = 0;
	int out_c = 0;
	int batch = 0;
	int outputs = 0;
	size_t workspace_size = 0;

	float* output = nullptr;
	float* delta = nullptr;

#ifdef GPU
	float* output_gpu = nullptr;
	float* delta_gpu = nullptr;
#endif
};

}

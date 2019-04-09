
#pragma once

#include "include/darknet.h"

namespace darknet {

struct Network;

struct Layer: layer {
	Layer(): layer() {

	}

	virtual ~Layer() {};

	Layer(const Layer& other) = delete;
	Layer& operator=(const Layer& other) = delete;

	virtual void load_weights(FILE* fp) = 0;
	virtual void save_weights(FILE* fp) const = 0;

	virtual void forward(Network* network) = 0;
	virtual void backward(Network* network) = 0;
	virtual void update(Network* network) = 0;
#ifdef GPU
	virtual void forward_gpu(Network* network) = 0;
	virtual void backward_gpu(Network* network) = 0;
	virtual void update_gpu(Network* network) = 0;
#endif
};

}

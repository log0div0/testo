
#pragma once

#include "../Layer.hpp"
#include <inipp.hh>

namespace darknet {

struct MaxPoolLayer: Layer {
	MaxPoolLayer(const inipp::inisection& section,
		size_t batch,
		size_t w,
		size_t h,
		size_t c);
	virtual ~MaxPoolLayer() override;

	virtual void load_weights(FILE* fp) override {};
	virtual void save_weights(FILE* fp) const override {};

	virtual void forward(Network* network) override;
	virtual void backward(Network* network) override;
	virtual void update(Network* network) override {};
#ifdef GPU
	virtual void forward_gpu(Network* network) override;
	virtual void backward_gpu(Network* network) override;
	virtual void update_gpu(Network* network) override {};
#endif

	int in_w = 0;
	int in_h = 0;
	int in_c = 0;
	int stride_w = 0;
	int stride_h = 0;
	int size_w = 0;
	int size_h = 0;
	int pad_w = 0;
	int pad_h = 0;

	int* indexes = nullptr;

#ifdef GPU
	int* indexes_gpu = nullptr;
#endif
};

}

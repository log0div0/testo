
#pragma once

#include "../Layer.hpp"
#include <inipp.hh>

namespace darknet {

struct ConvolutionalLayer: Layer {
	ConvolutionalLayer(const inipp::inisection& section,
		size_t batch,
		size_t w,
		size_t h,
		size_t c);
	virtual ~ConvolutionalLayer() override;

	virtual void load_weights(FILE* fp) override;
	virtual void save_weights(FILE* fp) const override;


	virtual void forward(Network* network) override;
	virtual void backward(Network* network) override;
	virtual void update(Network* network) override;
#ifdef GPU
	virtual void forward_gpu(Network* network) override;
	virtual void backward_gpu(Network* network) override;
	virtual void update_gpu(Network* network) override;
#endif

private:
	size_t get_workspace_size() const;
	int get_out_height() const;
	int get_out_width() const;

#ifdef GPU
	void pull() const;
	void push() const;
#endif
};

}

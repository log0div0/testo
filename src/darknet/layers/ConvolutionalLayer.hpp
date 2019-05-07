
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

	virtual void load_weights(std::istream& stream) override;
	virtual void save_weights(FILE* fp) const override;


	virtual void forward(Network* network) override;
	virtual void backward(Network* network) override;
	virtual void update(Network* network, float learning_rate, float momentum, float decay) override;
#ifdef GPU
	virtual void forward_gpu(Network* network) override;
	virtual void backward_gpu(Network* network) override;
	virtual void update_gpu(Network* network, float learning_rate, float momentum, float decay) override;
#endif

private:
	void im2col_cpu(float* data_im, float* data_col);
	void col2im_cpu(float* data_col, float* data_im);

	size_t get_workspace_size() const;
	int get_out_height() const;
	int get_out_width() const;

#ifdef GPU
	void im2col_gpu(float* data_im, float* data_col);
	void col2im_gpu(float* data_col, float* data_im);

	void pull() const;
	void push() const;
#endif

	int in_w = 0;
	int in_h = 0;
	int in_c = 0;
	int size = 0;
	int stride = 0;
	int pad = 0;
	bool batch_normalize = false;
	ACTIVATION activation;

	float* weights = nullptr;
	float* weight_updates = nullptr;

	float* biases = nullptr;
	float* bias_updates = nullptr;

	int nweights = 0;
	int nbiases = 0;

	float* scales = nullptr;
	float* scale_updates = nullptr;

	float* mean = nullptr;
	float* variance = nullptr;

	float* mean_delta = nullptr;
	float* variance_delta = nullptr;

	float* rolling_mean = nullptr;
	float* rolling_variance = nullptr;

	float* x = nullptr;
	float* x_norm = nullptr;

#ifdef GPU
	float* weights_gpu = nullptr;
	float* weight_updates_gpu = nullptr;

	float* biases_gpu = nullptr;
	float* bias_updates_gpu = nullptr;

	float* scales_gpu = nullptr;
	float* scale_updates_gpu = nullptr;

	float* mean_gpu = nullptr;
	float* variance_gpu = nullptr;

	float* mean_delta_gpu = nullptr;
	float* variance_delta_gpu = nullptr;

	float* rolling_mean_gpu = nullptr;
	float* rolling_variance_gpu = nullptr;

	float* x_gpu = nullptr;
	float* x_norm_gpu = nullptr;
#endif
};

}

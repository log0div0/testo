
#pragma once

#include <string>
#include <vector>
#include <memory>
#include "Image.hpp"
#include "Layer.hpp"

namespace darknet {

struct Network {
	Network(const std::string& config_file_path);
	~Network();

	Network(const Network&) = delete;
	Network& operator=(const Network&) = delete;

	Network(Network&&);
	Network& operator=(Network&&);

	void load_weights(const std::string& weights_file_path);
	void save_weights(const std::string& weights_file_path);

	void forward();
	void backward();
	void update();

	std::vector<std::unique_ptr<Layer>> layers;

	int batch = 0;

	float learning_rate = 0;
	float momentum = 0;
	float decay = 0;

	int inputs = 0;
	int outputs = 0;
	int h = 0, w = 0, c = 0;


	float *input = nullptr;
	float *delta = nullptr;
	float *workspace = nullptr;
	bool train = false;

#ifdef GPU
	float *input_gpu = nullptr;
	float *truth_gpu = nullptr;
	float *delta_gpu = nullptr;
	float *output_gpu = nullptr;
#endif

};

}

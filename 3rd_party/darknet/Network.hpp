
#pragma once

#include <string>
#include <vector>
#include "include/darknet.h"
#include "Image.hpp"

namespace darknet {

struct Network: network {
	Network(const std::string& config_file_path
#ifdef GPU
		, int gpu = -1
#endif
	);
	~Network();

	Network(const Network&) = delete;
	Network& operator=(const Network&) = delete;

	Network(Network&&);
	Network& operator=(Network&&);

	void load_weights(const std::string& weights_file_path);
	void save_weights(const std::string& weights_file_path);
	void set_batch(size_t batch);
	float* predict(const Image& image);

	const layer& back() const {
		return layers[n-1];
	}

	size_t width() const {
		return w;
	}

	size_t height() const {
		return h;
	}

private:
	float* forward(float* input);
};

}

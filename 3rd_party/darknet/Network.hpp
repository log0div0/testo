
#pragma once

#include <string>
#include "include/darknet.h"
#include "Image.hpp"

namespace darknet {

struct Network {
	Network(const std::string& config_file_path);
	~Network();

	Network(const Network&) = delete;
	Network& operator=(const Network&) = delete;

	void load_weights(const std::string& weights_file_path);
	void set_batch(size_t batch);
	float* predict(const Image& image);

	const layer& back() const {
		return impl->layers[impl->n-1];
	}

	size_t width() const {
		return impl->w;
	}

	size_t height() const {
		return impl->h;
	}

	network* impl = nullptr;

private:
	float* forward(float* input);
};

}


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
	void set_batch(int batch);
	float* predict(const Image& image);

	const layer& back() const {
		return impl->layers[impl->n-1];
	}

	int width() const {
		return impl->w;
	}

	int height() const {
		return impl->h;
	}

private:
	float* forward(float* input);
	network* impl = nullptr;
};

}

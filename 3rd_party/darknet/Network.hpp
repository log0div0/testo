
#pragma once

#include <string>
#include <vector>
#include "include/darknet.h"
#include "Image.hpp"

namespace darknet {

struct Network: network {
	Network(const std::string& config_file_path);
	~Network();

	Network(const Network&) = delete;
	Network& operator=(const Network&) = delete;

	Network(Network&&);
	Network& operator=(Network&&);

	void load_weights(const std::string& weights_file_path);
	void save_weights(const std::string& weights_file_path);

	const layer& back() const {
		return layers[n-1];
	}

	size_t width() const {
		return w;
	}

	size_t height() const {
		return h;
	}

	void forward();
	void backward();
	void update();
};

}

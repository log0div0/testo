
#pragma once

#include <string>
#include <vector>
#include <memory>
#include "Image.hpp"
#include "Layer.hpp"

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

	void forward();
	void backward();
	void update();

	std::vector<std::unique_ptr<Layer>> layers;
};

}

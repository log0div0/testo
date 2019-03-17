
#pragma once

#include "Network.hpp"

namespace darknet {

struct Trainer {
	Trainer(const std::string& config_file_path, const std::vector<int> gpus);

	void load_weights(const std::string& weights_file_path);
	void sync_weights();
	void save_weights(const std::string& weights_file_path);

	size_t max_batches() const;
	size_t current_batch() const;
	size_t batch_size() const;
	size_t subdivisions() const;

	float train(data data);

	std::vector<Network> networks;
};

}

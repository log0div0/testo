
#pragma once

#include "Network.hpp"
#include "Dataset.hpp"

namespace darknet {

struct Trainer {
	Trainer(const std::string& network_file, const std::string& dataset_file
#ifdef GPU
		, const std::vector<int> gpus
#endif
	);

	void load_weights(const std::string& weights_file_path);
	void sync_weights();
	void save_weights(const std::string& weights_file_path);

	float train();

	size_t current_batch() const;

#ifdef GPU
	std::vector<Network> networks;
#else
	Network network;
#endif
	Dataset dataset;
};

}

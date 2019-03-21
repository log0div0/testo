
#pragma once

#include "Network.hpp"
#include "Dataset.hpp"

namespace darknet {

struct Trainer {
	Trainer(const std::string& network_file, const std::string& dataset_file,
#ifdef GPU
		const std::vector<int> gpus
#else
		size_t threads_count
#endif
	);

	void load_weights(const std::string& weights_file_path);
	void sync_weights();
	void save_weights(const std::string& weights_file_path);

	float train();

	std::vector<Network> networks;
	Dataset dataset;
	size_t batch_index = 0;

private:
	float train(Network& network);
};

}

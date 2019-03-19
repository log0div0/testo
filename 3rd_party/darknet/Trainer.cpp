
#include "Trainer.hpp"

namespace darknet {

Trainer::Trainer(const std::string& network_file, const std::string& dataset_file
#ifdef GPU
	, const std::vector<int> gpus
#endif
):
#ifndef GPU
	network(network_file),
#endif
	dataset(dataset_file)
{
#ifdef GPU
	int seed = rand();
	for (size_t gpu: gpus) {
		cuda_set_device(gpu);
		srand(seed);
		networks.push_back(Network(network_file));
		networks.back().impl->gpu_index = gpu;
		networks.back().impl->learning_rate *= gpus.size();
	}
#endif
}

void Trainer::load_weights(const std::string& weights_file_path) {
#ifdef GPU
	for (auto& network: networks) {
		network.load_weights(weights_file_path);
	}
#else
	network.load_weights(weights_file_path);
#endif
}

void Trainer::sync_weights() {
#ifdef GPU
	if (networks.size() != 1) {
		std::vector<network*> n;
		for (auto& network: networks) {
			n.push_back(network.impl);
		}
		sync_nets(n.data(), n.size(), 0);
	}
#endif
}

void Trainer::save_weights(const std::string& weights_file_path) {
#ifdef GPU
	sync_weights();
	networks.at(0).save_weights(weights_file_path);
#else
	network.save_weights(weights_file_path);
#endif
}

float Trainer::train() {
#ifdef GPU
	Data data = dataset.load(networks.at(0).impl->batch * networks.at(0).impl->subdivisions * networks.size());
	if (networks.size() == 1) {
		return train_network(networks.back().impl, data);
	} else {
		std::vector<network*> n;
		for (auto& network: networks) {
			n.push_back(network.impl);
		}
		return ::train_networks(n.data(), n.size(), data, 4);
	}
#else
	Data data = dataset.load(network.impl->batch * network.impl->subdivisions);
	return train_network(network.impl, data);
#endif
}

size_t Trainer::current_batch() const {
#ifdef GPU
	return get_current_batch(networks.at(0).impl);
#else
	return get_current_batch(network.impl);
#endif
}

}

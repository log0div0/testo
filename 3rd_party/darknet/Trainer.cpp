
#include "Trainer.hpp"

namespace darknet {

Trainer::Trainer(const std::string& config_file_path, const std::vector<int> gpus) {
	int seed = rand();
	for (size_t gpu: gpus) {
		srand(seed);
		networks.push_back(Network(config_file_path));
		networks.back().impl->gpu_index = gpu;
		networks.back().impl->learning_rate *= gpus.size();
	}
}

void Trainer::load_weights(const std::string& weights_file_path) {
	for (auto& network: networks) {
		network.load_weights(weights_file_path);
	}
}

void Trainer::sync_weights() {
#ifdef GPU
	if (networks.size() != 1) {
		std::vector<network*> n(networks.size());
		for (auto& network: networks) {
			n.push_back(network.impl);
		}
		sync_nets(n.data(), n.size(), 0);
	}
#endif
}

void Trainer::save_weights(const std::string& weights_file_path) {
	sync_weights();
	networks.at(0).save_weights(weights_file_path);
}

float Trainer::train(data data) {
#ifdef GPU
	if (networks.size() == 1) {
		return train_network(networks.back().impl, data);
	} else {
		std::vector<network*> n(networks.size());
		for (auto& network: networks) {
			n.push_back(network.impl);
		}
		return ::train_networks(n.data(), n.size(), data, 4);
	}
#else
	return train_network(networks.back().impl, data);
#endif
}

size_t Trainer::max_batches() const {
	return networks.at(0).impl->max_batches;
}

size_t Trainer::current_batch() const {
	return get_current_batch(networks.at(0).impl);
}

size_t Trainer::batch_size() const {
	return networks.at(0).impl->batch;
}

size_t Trainer::subdivisions() const {
	return networks.at(0).impl->subdivisions;
}

}

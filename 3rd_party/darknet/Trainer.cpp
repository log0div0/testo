
#include "Trainer.hpp"
#include <future>

extern "C" {
void merge_weights(layer l, layer base);
void scale_weights(layer l, float s);
}

namespace darknet {

Trainer::Trainer(const std::string& network_file, const std::string& dataset_file,
#ifdef GPU
	const std::vector<int> gpus
#else
	size_t threads_count
#endif
):
	dataset(dataset_file)
{
	int seed = rand();
#ifdef GPU
	for (int gpu: gpus) {
		srand(seed);
		networks.push_back(Network(network_file, gpu));
	}
#else
	for (size_t i = 0; i < threads_count; ++i) {
		srand(seed);
		networks.push_back(Network(network_file));
	}
#endif
}

void Trainer::load_weights(const std::string& weights_file_path) {
	for (auto& network: networks) {
		network.load_weights(weights_file_path);
	}
}

void Trainer::sync_weights() {
	if (networks.size() == 1) {
		return;
	}
	for (size_t j = 0; j < networks[0].n; ++j) {
		layer base = networks[0].layers[j];
#ifdef GPU
		cuda_set_device(networks[0].gpu_index);
		pull_weights(base);
#endif
		for (size_t i = 1; i < networks.size(); ++i) {
			layer l = networks[i].layers[j];
#ifdef GPU
			cuda_set_device(networks[i].gpu_index);
			pull_weights(l);
#endif
			merge_weights(l, base);
		}
		scale_weights(base, 1./networks.size());
		for (size_t i = 1; i < networks.size(); ++i) {
			layer l = networks[i].layers[j];
			scale_weights(l, 0);
			merge_weights(base, l);
#ifdef GPU
			cuda_set_device(networks[i].gpu_index);
			push_weights(l);
#endif
		}
	}
}

void Trainer::save_weights(const std::string& weights_file_path) {
	sync_weights();
	networks.at(0).save_weights(weights_file_path);
}

float Trainer::train() {

	std::vector<std::future<float>> errors;
	for (size_t i = 0; i < networks.size(); ++i) {
		errors.push_back(std::async(std::launch::async, [=] {
			return train(networks[i]);
		}));
	}
	float sum = 0;
	for (auto& error: errors) {
		sum += error.get();
	}

	if (++batch_index % 4 == 0) {
		sync_weights();
	}

	return sum / networks.size();
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        if(y) memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

float Trainer::train(Network& network) {
	Data d = dataset.load(network.batch);
	network.train = 1;
	get_next_batch(d, d.X.rows, 0, network.input, network.truth);
    forward_network(&network);
    backward_network(&network);
    float sum = 0;
    int count = 0;
    for(size_t i = 0; i < network.n; ++i) {
        if (network.layers[i].cost) {
            sum += network.layers[i].cost[0];
            ++count;
        }
    }
    float error = sum/count;
    update_network(&network);
    return error;
}

}

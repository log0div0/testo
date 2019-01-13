
#include "Network.hpp"

namespace darknet {

Network::Network(const std::string& config_file_name) {
	impl = parse_network_cfg((char*)config_file_name.c_str());
	if (!impl) {
		throw std::runtime_error("parse_network_cfg failed");
	}
}

Network::~Network() {
	if (impl) {
		free_network(impl);
		impl = nullptr;
	}
}

void Network::load_weights(const std::string& weights_file_path) {
	::load_weights(impl, (char*)weights_file_path.c_str());
}

void Network::set_batch(int batch) {
	set_batch_network(impl, 1);
}

float* Network::predict(const Image& image, float thresh) {
	Image resized_image = image.letterbox(impl->w, impl->h);
	return forward(resized_image.data());
}

float* Network::forward(float* in) {
	network orig = *impl;
	impl->input = in;
	impl->truth = 0;
	impl->train = 0;
	impl->delta = 0;
	forward_network(impl);
	float *out = impl->output;
	*impl = orig;
	return out;
}

}
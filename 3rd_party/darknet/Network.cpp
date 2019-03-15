
#include "Network.hpp"
#include <stdexcept>

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

void Network::set_batch(size_t batch) {
	set_batch_network(impl, batch);
}

float* Network::predict(const Image& image) {
	if ((width() != image.width()) ||
		(height() != image.height()))
	{
		return forward(image.letterbox(width(), height()).data());
	} else {
		return forward((float*)image.data());
	}
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
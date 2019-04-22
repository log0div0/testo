
#include "StinkingPileOfShit.hpp"
#include <darknet/YOLO.hpp>

bool StinkingPileOfShit::stink_even_stronger(stb::Image& image, const std::string& text) {
	if (network) {
		if ((network->w != image.width) ||
			(network->h != image.height) ||
			(network->c != image.channels)) {
			network.reset();
		}
	}
	if (!network) {
		use_gpu = true;
		network = std::make_unique<darknet::Network>("/home/log0div0/work/testo/nn/testo.network",
			1, image.width, image.height, image.channels);
		network->load_weights("/home/log0div0/work/testo/nn/testo.weights");
	}

	return yolo::predict(*network, image, text);
}
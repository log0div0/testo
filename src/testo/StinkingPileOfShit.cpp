
#include "StinkingPileOfShit.hpp"
#include <darknet/YOLO.hpp>

extern unsigned char testo_network[];
extern unsigned int testo_network_len;
extern unsigned char testo_weights[];
extern unsigned int testo_weights_len;

struct membuf: std::basic_streambuf<char> {
	membuf(const uint8_t *p, size_t l) {
		setg((char*)p, (char*)p, (char*)p + l);
	}
};

StinkingPileOfShit::StinkingPileOfShit() {
	config = std::string(testo_network, testo_network + testo_network_len);
}

bool StinkingPileOfShit::stink_even_stronger(stb::Image& image, const std::string& text) {
	if (network) {
		if ((network->w != image.width) ||
			(network->h != image.height) ||
			(network->c != image.channels)) {
			network.reset();
		}
	}
	if (!network) {
		std::stringstream ss(config);
		use_gpu = true;
		network = std::make_unique<darknet::Network>(ss, 1, image.width, image.height, image.channels);
		membuf buf(testo_weights, testo_weights_len);
		std::istream bs(&buf);
		network->load_weights(bs);
	}

	return yolo::predict(*network, image, text);
}

#pragma once

#include <darknet/Image.hpp>
#include <darknet/Network.hpp>

struct StinkingPileOfShit {

	bool stink_even_stronger(stb::Image& image, const std::string& text);

private:
	std::unique_ptr<darknet::Network> network;
};

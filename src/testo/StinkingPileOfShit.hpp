
#pragma once

#include <darknet/Image.hpp>
#include <darknet/Network.hpp>
#include <inipp.hh>

struct StinkingPileOfShit {
	StinkingPileOfShit();
	bool stink_even_stronger(stb::Image& image, const std::string& text,
		const std::string& foreground,
		const std::string& background);

private:
	std::string config;
	std::map<std::string, int> symbols;
	std::vector<uint8_t> weights;
	std::unique_ptr<darknet::Network> network;
};


#include "HyperVWinChannel.hpp"
#include <stdexcept>

HyperVWinChannel::HyperVWinChannel() {
	throw std::runtime_error("Not implemented");
}
HyperVWinChannel::~HyperVWinChannel() {

}

HyperVWinChannel::HyperVWinChannel(HyperVWinChannel&& other) {
	throw std::runtime_error("Not implemented");
}

HyperVWinChannel& HyperVWinChannel::operator=(HyperVWinChannel&& other) {
	throw std::runtime_error("Not implemented");
}

size_t HyperVWinChannel::read(uint8_t* data, size_t size) {
	throw std::runtime_error("Not implemented");
}

size_t HyperVWinChannel::write(uint8_t* data, size_t size) {
	throw std::runtime_error("Not implemented");
}

void HyperVWinChannel::close() {
	throw std::runtime_error("Not implemented");
}

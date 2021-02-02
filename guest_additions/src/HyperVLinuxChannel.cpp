
#include "HyperVLinuxChannel.hpp"
#include <stdexcept>

HyperVLinuxChannel::HyperVLinuxChannel() {
	throw std::runtime_error("Not implemented");
}
HyperVLinuxChannel::~HyperVLinuxChannel() {

}

HyperVLinuxChannel::HyperVLinuxChannel(HyperVLinuxChannel&& other) {
	throw std::runtime_error("Not implemented");
}

HyperVLinuxChannel& HyperVLinuxChannel::operator=(HyperVLinuxChannel&& other) {
	throw std::runtime_error("Not implemented");
}

size_t HyperVLinuxChannel::read(uint8_t* data, size_t size) {
	throw std::runtime_error("Not implemented");
}

size_t HyperVLinuxChannel::write(uint8_t* data, size_t size) {
	throw std::runtime_error("Not implemented");
}

void HyperVLinuxChannel::close() {
	throw std::runtime_error("Not implemented");
}

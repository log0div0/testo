
#include "HyperVChannel.hpp"
#include <stdexcept>
#include <spdlog/spdlog.h>

HyperVChannel::HyperVChannel(HyperVChannel&& other):
	socket(std::move(other.socket))
{

}

HyperVChannel& HyperVChannel::operator=(HyperVChannel&& other) {
	socket = std::move(other.socket);
	return *this;
}

size_t HyperVChannel::read(uint8_t* data, size_t size) {
	return socket.readSome(data, size);
}

size_t HyperVChannel::write(uint8_t* data, size_t size) {
	return socket.writeSome(data, size);
}

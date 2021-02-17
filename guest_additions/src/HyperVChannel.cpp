
#include "HyperVChannel.hpp"
#include <stdexcept>
#include <spdlog/spdlog.h>

#define HYPERV_PORT 1234
#ifdef WIN32
DEFINE_GUID(service_id, HYPERV_PORT, 0xfacb, 0x11e6, 0xbd, 0x58, 0x64, 0x00, 0x6a, 0x79, 0x86, 0xd3);
HyperVChannel::HyperVChannel(): acceptor(hyperv::VSocketEndpoint(service_id))
#else
HyperVChannel::HyperVChannel(): acceptor(hyperv::VSocketEndpoint(HYPERV_PORT))
#endif
{
	spdlog::info("Listening on hyperv port {}", HYPERV_PORT);
}

HyperVChannel::~HyperVChannel() {

}

HyperVChannel::HyperVChannel(HyperVChannel&& other):
	acceptor(std::move(other.acceptor)),
	socket(std::move(other.socket))
{

}

HyperVChannel& HyperVChannel::operator=(HyperVChannel&& other) {
	acceptor = std::move(other.acceptor);
	socket = std::move(other.socket);
	return *this;
}

size_t HyperVChannel::read(uint8_t* data, size_t size) {
	if (!socket) {
		socket.reset(new Socket(acceptor.accept()));
	}
	try {
		size_t n = socket->readSome(data, size);
		return n;
	} catch (const std::exception& error) {
		socket.reset();
		throw;
	}
}

size_t HyperVChannel::write(uint8_t* data, size_t size) {
	if (!socket) {
		socket.reset(new Socket(acceptor.accept()));
	}
	try {
		size_t n = socket->writeSome(data, size);
		return n;
	} catch (const std::exception& error) {
		socket.reset();
		throw;
	}
}

void HyperVChannel::close() {

}

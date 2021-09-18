
#pragma once

#define HYPERV_PORT 1234

#ifdef WIN32
#include <hyperv/AsioWin.hpp>
#else
#include <hyperv/AsioLinux.hpp>
#endif
#include "Channel.hpp"

struct HyperVChannel: Channel {
	using Socket = coro::StreamSocket<hyperv::VSocketProtocol>;

	HyperVChannel(Socket socket_): socket(std::move(socket_)) {}
	~HyperVChannel() = default;

	HyperVChannel(HyperVChannel&& other);
	HyperVChannel& operator=(HyperVChannel&& other);

	size_t read(uint8_t* data, size_t size) override;
	size_t write(uint8_t* data, size_t size) override;

private:
	Socket socket;
};

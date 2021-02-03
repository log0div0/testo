
#pragma once

#ifdef WIN32
#include <hyperv/AsioWin.hpp>
#else
#include <hyperv/AsioLinux.hpp>
#endif

struct HyperVChannel {
	HyperVChannel();
	~HyperVChannel();

	HyperVChannel(HyperVChannel&& other);
	HyperVChannel& operator=(HyperVChannel&& other);

	size_t read(uint8_t* data, size_t size);
	size_t write(uint8_t* data, size_t size);
	void close();

private:
	using Acceptor = coro::Acceptor<hyperv::VSocketProtocol>;
	using Socket = coro::StreamSocket<hyperv::VSocketProtocol>;

	Acceptor acceptor;
	std::unique_ptr<Socket> socket;
};

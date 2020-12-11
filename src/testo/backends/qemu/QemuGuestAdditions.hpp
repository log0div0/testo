
#pragma once

#include "../GuestAdditions.hpp"
#include "qemu/Domain.hpp"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic push
#include <coro/StreamSocket.h>
#include <coro/Timeout.h>
#pragma GCC diagnostic pop

struct QemuGuestAdditions: GuestAdditions {
	QemuGuestAdditions(vir::Domain& domain);

private:
	virtual void send_raw(const uint8_t* data, size_t size) override;
	virtual void recv_raw(uint8_t* data, size_t size) override;

	using Socket = coro::StreamSocket<asio::local::stream_protocol>;
	using Endpoint = asio::local::stream_protocol::endpoint;

	Socket socket;
	Endpoint endpoint;
};

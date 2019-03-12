
#pragma once

#include "qemu/Domain.hpp"
#include <coro/StreamSocket.h>
#include <coro/Timeout.h>
#include <nlohmann/json.hpp>

struct Negotiator {
	Negotiator() = delete;
	Negotiator(const Negotiator& other) = delete;
	Negotiator& operator=(const Negotiator& other) = delete;

	Negotiator(vir::Domain& domain);

	bool is_avaliable();

private:
	using Socket = coro::StreamSocket<asio::local::stream_protocol>;
	using Endpoint = asio::local::stream_protocol::endpoint;

	void send(const nlohmann::json& command);
	nlohmann::json recv();

	Socket socket;
	Endpoint endpoint;
};



#pragma once

#include "qemu/Domain.hpp"
#include <coro/StreamSocket.h>
#include <coro/Timeout.h>
#include "Utils.hpp"
#include <nlohmann/json.hpp>

struct Negotiator {
	Negotiator() = delete;
	Negotiator(const Negotiator& other) = delete;
	Negotiator& operator=(const Negotiator& other) = delete;

	Negotiator(vir::Domain& domain);

	bool is_avaliable();
	void copy_to_guest(const fs::path& src, const fs::path& dst);
	int execute(const std::string& command);

private:
	using Socket = coro::StreamSocket<asio::local::stream_protocol>;
	using Endpoint = asio::local::stream_protocol::endpoint;

	void copy_file_to_guest(const fs::path& src, const fs::path& dst);
	void copy_dir_to_guest(const fs::path& src, const fs::path& dst);

	void send(const nlohmann::json& command);
	nlohmann::json recv();

	Socket socket;
	Endpoint endpoint;
};



#pragma once

#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic push
#include <coro/StreamSocket.h>
#include <coro/Timeout.h>
#pragma GCC diagnostic pop
#include "../../Utils.hpp"
#include <nlohmann/json.hpp>

struct QemuChannelHandler {
	virtual ~QemuChannelHandler() = default;

	virtual void send(const nlohmann::json& command) = 0;
	virtual nlohmann::json recv() = 0;
};

struct QemuUnixChannelHandler: public QemuChannelHandler {
	QemuUnixChannelHandler(const fs::path& sock_path);

	void send(const nlohmann::json& command) override;
	nlohmann::json recv() override;

private:
	using Socket = coro::StreamSocket<asio::local::stream_protocol>;
	using Endpoint = asio::local::stream_protocol::endpoint;

	Socket socket;
	Endpoint endpoint;
};

struct QemuTCPChannelHandler: public QemuChannelHandler {
	QemuTCPChannelHandler(const std::string& remote_host, const std::string& port);

	void send(const nlohmann::json& command) override;
	nlohmann::json recv() override;

private:
	using Socket = coro::TcpSocket;

	Socket socket;
};

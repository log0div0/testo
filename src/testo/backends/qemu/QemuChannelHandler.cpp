
#include "QemuChannelHandler.hpp"
#include "coro/Resolver.h"

QemuUnixChannelHandler::QemuUnixChannelHandler(const fs::path& sock_path) {
	endpoint = Endpoint(sock_path);
	socket.connect(endpoint);
}


void QemuUnixChannelHandler::send(const nlohmann::json& command) {
	auto command_str = command.dump();
	std::vector<uint8_t> buffer;
	uint32_t command_length = command_str.length();
	buffer.reserve(sizeof(uint32_t) + command_str.length());
	std::copy((uint8_t*)&command_length, (uint8_t*)(&command_length) + sizeof(uint32_t), std::back_inserter(buffer));

	std::copy(command_str.begin(), command_str.end(), std::back_inserter(buffer));
	socket.write(buffer);
}

nlohmann::json QemuUnixChannelHandler::recv() {
	uint32_t json_length = 0;
	socket.read(&json_length, sizeof(uint32_t));
	std::string json_str;
	json_str.resize(json_length);
	socket.read(&json_str[0], json_length);
	return nlohmann::json::parse(json_str);
}

QemuTCPChannelHandler::QemuTCPChannelHandler(const std::string& remote_host, const std::string& port) {
	coro::TcpResolver resolver;
	asio::ip::tcp::resolver::query resolver_query(remote_host, port, asio::ip::tcp::resolver::query::numeric_service);
	auto it = resolver.resolve(resolver_query);
	socket.connect(it->endpoint());
}


void QemuTCPChannelHandler::send(const nlohmann::json& command) {
	auto command_str = command.dump();
	std::vector<uint8_t> buffer;
	uint32_t command_length = command_str.length();
	buffer.reserve(sizeof(uint32_t) + command_str.length());
	std::copy((uint8_t*)&command_length, (uint8_t*)(&command_length) + sizeof(uint32_t), std::back_inserter(buffer));

	std::copy(command_str.begin(), command_str.end(), std::back_inserter(buffer));
	socket.write(buffer);
}

nlohmann::json QemuTCPChannelHandler::recv() {
	uint32_t json_length = 0;
	socket.read(&json_length, sizeof(uint32_t));
	std::string json_str;
	json_str.resize(json_length);
	socket.read(&json_str[0], json_length);
	return nlohmann::json::parse(json_str);
}
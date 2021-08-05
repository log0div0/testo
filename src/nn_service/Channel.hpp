
#pragma once

#include "coro/StreamSocket.h"
#include <thread>
#include <chrono>
#include <memory>
#include <iostream>

#include "Messages.hpp"
#include <nlohmann/json.hpp>

using Socket = coro::StreamSocket<asio::ip::tcp>;
using Endpoint = asio::ip::tcp::endpoint;

struct Channel {
	Channel(Socket _socket): socket(std::move(_socket)) {}
	~Channel() = default;

	nlohmann::json recv();
	void send(const nlohmann::json& message);

	Socket socket;
};


inline nlohmann::json Channel::recv() {	
	uint32_t msg_size;

	socket.read((uint8_t*)&msg_size, 4);

	std::vector<uint8_t> json_data;
	json_data.resize(msg_size);
	socket.read((uint8_t*)json_data.data(), json_data.size());

	return nlohmann::json::from_cbor(json_data);
}

inline void Channel::send(const nlohmann::json& json) {
	std::vector<uint8_t> json_data = nlohmann::json::to_cbor(json);
	uint32_t json_size = (uint32_t)json_data.size();
	socket.write((uint8_t*)&json_size, sizeof(json_size));
	socket.write((uint8_t*)json_data.data(), json_size);
}

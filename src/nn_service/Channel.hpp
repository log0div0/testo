
#pragma once

#include "coro/StreamSocket.h"
#include <thread>
#include <chrono>
#include <memory>

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

	/*ImageSize screenshot_size = header["screenshot"].get<ImageSize>();

	if (screenshot_size.c != 3) {
		throw std::runtime_error("Unsupported channel number");
	}

	std::unique_ptr<Message> result;

	if (!header.count("type")) {
		throw std::runtime_error("The request doesn't have the \"type\" field");
	}

	auto type = header["type"].get<std::string>();

	if (type == "js") {
		result.reset(new JSRequest());
	} else if (type == "ref_image") {
		result.reset(new RefImage());
	} else {
		throw std::runtime_error("Uknown request type: " + type);
	}

	result->header = header;

	result->screenshot = stb::Image<stb::RGB>(screenshot_size.w, screenshot_size.h);

	socket.read(result->screenshot.data, screenshot_size.total_size());

	if (auto p = dynamic_cast<JSRequest*>(result.get())) {
		auto script_size = header.at("js_size").get<uint32_t>();
		p->script.resize(script_size);
		socket.read((uint8_t*)p->script.data(), script_size);
	}

	return result;*/
}

inline void Channel::send(const nlohmann::json& json) {
	std::vector<uint8_t> json_data = nlohmann::json::to_cbor(json);
	uint32_t json_size = (uint32_t)json_data.size();
	socket.write((uint8_t*)&json_size, sizeof(json_size));
	socket.write((uint8_t*)json_data.data(), json_size);
}

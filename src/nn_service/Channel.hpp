
#pragma once
#include "Messages.hpp"

#include "coro/StreamSocket.h"
#include <thread>
#include <chrono>

#include <memory>

using Socket = coro::StreamSocket<asio::ip::tcp>;
using Endpoint = asio::ip::tcp::endpoint;

struct Channel {
	Channel(Socket _socket): socket(std::move(_socket)) {}
	~Channel() = default;

	std::unique_ptr<Message> receive_message();
	void send_request(const JSRequest& msg);
	void send_request(const RefImage& msg);

	void send_response(const nlohmann::json& response);
	nlohmann::json receive_response();

	Socket socket;
private:
	void send_request(const Message& msg);
	void send_json(const nlohmann::json& json);
};

inline std::unique_ptr<Message> Channel::receive_message() {	
	uint32_t header_size;

	socket.read((uint8_t*)&header_size, 4);

	std::string json_str;
	json_str.resize(header_size);
	socket.read((uint8_t*)json_str.data(), json_str.size());

	auto header = nlohmann::json::parse(json_str);

	ImageSize screenshot_size = header["screenshot"].get<ImageSize>();

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

	return result;
}

inline void Channel::send_request(const Message& msg) {
	send_json(msg.header);

	size_t pic_size = msg.screenshot.w * msg.screenshot.h * msg.screenshot.c;
	socket.write(msg.screenshot.data, pic_size);
}

inline void Channel::send_json(const nlohmann::json& json) {
	auto json_str = json.dump();

	uint32_t json_size = (uint32_t)json_str.size();
	socket.write((uint8_t*)&json_size, sizeof(json_size));
	socket.write((uint8_t*)json_str.data(), json_size);
}

inline void Channel::send_request(const JSRequest& msg) {	
	send_request(static_cast<Message>(msg));

	socket.write(msg.script.data(), msg.script.length());
}

inline void Channel::send_request(const RefImage& msg) {	
	send_request(static_cast<Message>(msg));
}


inline void Channel::send_response(const nlohmann::json& response) {
	return send_json(response);
}

inline nlohmann::json Channel::receive_response() {
	uint32_t response_size;
	socket.read((uint8_t*)&response_size, 4);

	std::string json_str;
	json_str.resize(response_size);
	socket.read((uint8_t*)json_str.data(), json_str.size());

	auto result = nlohmann::json::parse(json_str);
	return result;
}


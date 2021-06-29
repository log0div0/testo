
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

	std::unique_ptr<Request> receive_request();
	void send_request(const TextRequest& msg);
	void send_request(const ImgRequest& msg);

	void send_response(const nlohmann::json& response);
	nlohmann::json receive_response();

private:
	void send_request(const Request& msg);
	void send_json(const nlohmann::json& json);
	Socket socket;
};

inline std::unique_ptr<Request> Channel::receive_request() {	
	uint32_t header_size;

	socket.read((uint8_t*)&header_size, 4);

	std::cout << "Header size: " << header_size << std::endl;

	std::string json_str;
	json_str.resize(header_size);
	socket.read((uint8_t*)json_str.data(), json_str.size());

	auto header = nlohmann::json::parse(json_str);

	std::cout << "Header: " << std::endl;
	std::cout << header.dump(4) << std::endl;

	ImageSize screenshot_size = header["screenshot"].get<ImageSize>();

	if (screenshot_size.c != 3) {
		throw std::runtime_error("Unsupported channel number");
	}

	std::unique_ptr<Request> result;

	if (header["type"].get<std::string>() == "text") {
		result.reset(new TextRequest());
	} else if (header["type"].get<std::string>() == "img") {
		result.reset(new ImgRequest());
	}

	result->header = header;
	result->screenshot = stb::Image<stb::RGB>(screenshot_size.w, screenshot_size.h);

	socket.read(result->screenshot.data, screenshot_size.total_size());

	if (auto p = dynamic_cast<ImgRequest*>(result.get())) {
		ImageSize pattern_size = header["pattern"].get<ImageSize>();

		if (pattern_size.c != 3) {
			throw std::runtime_error("Unsupported channel number");
		}

		socket.read(p->pattern.data, pattern_size.total_size());
	}

	return result;
}

inline void Channel::send_request(const Request& msg) {
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

inline void Channel::send_request(const TextRequest& msg) {
	return send_request(static_cast<Request>(msg));
}

inline void Channel::send_request(const ImgRequest& msg) {	
	send_request(static_cast<Request>(msg));

	size_t pattern_size = msg.pattern.w * msg.pattern.h * msg.pattern.c;
	socket.write(msg.pattern.data, pattern_size);
}

inline void Channel::send_response(const nlohmann::json& response) {
	return send_json(response);
}

inline nlohmann::json Channel::receive_response() {
	uint32_t response_size;
	socket.read((uint8_t*)&response_size, 4);

	std::cout << "Response size: " << response_size << std::endl;

	std::string json_str;
	json_str.resize(response_size);
	socket.read((uint8_t*)json_str.data(), json_str.size());

	auto result = nlohmann::json::parse(json_str);
	return result;
}


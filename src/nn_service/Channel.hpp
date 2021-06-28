
#pragma once
#include "Messages.hpp"

#include "coro/StreamSocket.h"
#include <thread>
#include <chrono>

#include <memory>

struct Channel {
	std::unique_ptr<Request> receive_request();
	void send_request(const TextRequest& msg);
	void send_request(const ImgRequest& msg);

	void receive_raw(uint8_t* data, size_t size);
	void send_raw(uint8_t* data, size_t size);

	virtual size_t read(uint8_t* data, size_t size) = 0;
	virtual size_t write(uint8_t* data, size_t size) = 0;

private:
	void send_request(const Request& msg);
};

using Socket = coro::StreamSocket<asio::ip::tcp>;
using Endpoint = asio::ip::tcp::endpoint;

struct TCPChannel: Channel {
	TCPChannel(Socket socket_): socket(std::move(socket_)) {}
	~TCPChannel() = default;

	TCPChannel(TCPChannel&& other);
	TCPChannel& operator=(TCPChannel&& other);

	size_t read(uint8_t* data, size_t size) override {
		return socket.readSome(data, size);
	}

	size_t write(uint8_t* data, size_t size) override {
		return socket.writeSome(data, size);
	}

private:
	Socket socket;
};

inline std::unique_ptr<Request> Channel::receive_request() {	
	uint32_t header_size;
	while (true) {
		size_t bytes_read = read((uint8_t*)&header_size, 4);
		if (bytes_read == 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		} else if (bytes_read != 4) {
			throw std::runtime_error("Can't read msg size");
		} else {
			break;
		}
	}

	std::cout << "Header size: " << header_size << std::endl;

	std::string json_str;
	json_str.resize(header_size);
	receive_raw((uint8_t*)json_str.data(), json_str.size());

	auto header = nlohmann::json::parse(json_str);

	std::cout << "Header: " << std::endl;
	std::cout << header.dump(4) << std::endl;

	int w = header["screenshot"]["w"].get<int>();
	int h = header["screenshot"]["h"].get<int>();
	int c = header["screenshot"]["c"].get<int>();

	if (c != 3) {
		throw std::runtime_error("Unsupported channel number");
	}

	std::unique_ptr<Request> result;

	if (header["type"].get<std::string>() == "text") {
		result.reset(new TextRequest());
	} else if (header["type"].get<std::string>() == "img") {
		result.reset(new ImgRequest());
	}

	result->header = header;
	result->screenshot = stb::Image<stb::RGB>(w, h);

	int screenshot_size = w * h * c;
	receive_raw(result->screenshot.data, screenshot_size);

	if (auto p = dynamic_cast<ImgRequest*>(result.get())) {

		w = header["pattern"]["w"].get<int>();
		h = header["pattern"]["h"].get<int>();
		c = header["pattern"]["c"].get<int>();

		if (c != 3) {
			throw std::runtime_error("Unsupported channel number");
		}

		int pattern_size = w * h * c;
		receive_raw(p->pattern.data, pattern_size);
	}

	return result;
}

inline void Channel::send_request(const Request& msg) {
	auto header_str = msg.header.dump();

	uint32_t header_size = (uint32_t)header_str.size();
	send_raw((uint8_t*)&header_size, sizeof(header_size));
	send_raw((uint8_t*)header_str.data(), header_size);

	size_t pic_size = msg.screenshot.w * msg.screenshot.h * msg.screenshot.c;
	send_raw(msg.screenshot.data, pic_size);
}

inline void Channel::send_request(const TextRequest& msg) {
	return send_request(static_cast<Request>(msg));
}

inline void Channel::send_request(const ImgRequest& msg) {	
	send_request(static_cast<Request>(msg));

	size_t pattern_size = msg.pattern.w * msg.pattern.h * msg.pattern.c;
	send_raw(msg.pattern.data, pattern_size);
}


inline void Channel::receive_raw(uint8_t* data, size_t size) {
	size_t already_read = 0;
	while (already_read < size) {
		size_t n = read(&data[already_read], size - already_read);
		if (n == 0) {
			throw std::runtime_error("EOF while reading");
		}
		already_read += n;
	}
}

inline void Channel::send_raw(uint8_t* data, size_t size) {
	size_t already_send = 0;
	while (already_send < size) {
		size_t n = write(&data[already_send], size - already_send);
		if (n == 0) {
			throw std::runtime_error("EOF while writing");
		}
		already_send += n;
	}
}


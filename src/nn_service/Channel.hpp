
#pragma once

#include <nlohmann/json.hpp>
#include <stb/Image.hpp>

#include <thread>
#include <chrono>
#include <string>

struct Request {
	enum class Type {
		TEXT,
		IMG
	};

	Request() = default;
	Request(const stb::Image<stb::RGB>& screenshot, const std::string& text_to_find);
	Request(const stb::Image<stb::RGB>& screenshot, const stb::Image<stb::RGB>& pattern);

	Type type() const {
		auto t = header["type"].get<std::string>();

		if (t == "text") {
			return Type::TEXT;
		} else if (t == "img") {
			return Type::IMG;
		} else {
			throw std::runtime_error("Unknown request type");
		}
	}

	std::string text_to_find() const {
		if (type() != Type::TEXT) {
			throw std::runtime_error("Can't get text_to_find field in no-txt request");
		}

		return header["text_to_find"].get<std::string>();
	}

	nlohmann::json header;
	stb::Image<stb::RGB> screenshot;
	stb::Image<stb::RGB> pattern;

private:
	void update_header(const std::string& field, const stb::Image<stb::RGB>& pic);
};

void Request::update_header(const std::string& field, const stb::Image<stb::RGB>& pic) {
	header[field] = nlohmann::json::object();

	header[field]["w"] = pic.w;
	header[field]["h"] = pic.h;
	header[field]["c"] = pic.c;
}

Request::Request(const stb::Image<stb::RGB>& screenshot, const std::string& text_to_find): screenshot(screenshot) {
	header["version"] = NN_SERVICE_PROCOTOL_VERSION;
	header["type"] = "text";
	header["text_to_find"] = text_to_find;
	update_header("screenshot", screenshot);
}

Request::Request(const stb::Image<stb::RGB>& screenshot, const stb::Image<stb::RGB>& pattern): screenshot(screenshot), pattern(pattern)
{
	header["version"] = NN_SERVICE_PROCOTOL_VERSION;
	header["type"] = "img";
	update_header("screenshot", screenshot);
	update_header("pattern", pattern);
}

struct Channel {
	Request receive_request();
	void send_request(const Request& msg);

	void receive_raw(uint8_t* data, size_t size);
	void send_raw(uint8_t* data, size_t size);

	virtual size_t read(uint8_t* data, size_t size) = 0;
	virtual size_t write(uint8_t* data, size_t size) = 0;
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

Request Channel::receive_request() {
	Request result;
	
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

	result.header = nlohmann::json::parse(json_str);

	std::cout << "Header: " << std::endl;
	std::cout << result.header.dump(4) << std::endl;

	int w = result.header["screenshot"]["w"].get<int>();
	int h = result.header["screenshot"]["h"].get<int>();
	int c = result.header["screenshot"]["c"].get<int>();

	if (c != 3) {
		throw std::runtime_error("Unsupported channel number");
	}

	result.screenshot = stb::Image<stb::RGB>(w, h);

	int screenshot_size = w * h * c;
	receive_raw(result.screenshot.data, screenshot_size);

	if (result.type() == Request::Type::IMG) {
		w = result.header["pattern"]["w"].get<int>();
		h = result.header["pattern"]["h"].get<int>();
		c = result.header["pattern"]["c"].get<int>();

		if (c != 3) {
			throw std::runtime_error("Unsupported channel number");
		}

		int pattern_size = w * h * c;
		receive_raw(result.pattern.data, pattern_size);
	}

	return result;
}

void Channel::send_request(const Request& msg) {
	auto header_str = msg.header.dump();

	uint32_t header_size = (uint32_t)header_str.size();
	send_raw((uint8_t*)&header_size, sizeof(header_size));
	send_raw((uint8_t*)header_str.data(), header_size);

	size_t pic_size = msg.screenshot.w * msg.screenshot.h * msg.screenshot.c;
	send_raw(msg.screenshot.data, pic_size);

	if (msg.type() == Request::Type::IMG) {
		size_t pattern_size = msg.pattern.w * msg.pattern.h * msg.pattern.c;
		send_raw(msg.pattern.data, pattern_size);
	}
}

void Channel::receive_raw(uint8_t* data, size_t size) {
	size_t already_read = 0;
	while (already_read < size) {
		size_t n = read(&data[already_read], size - already_read);
		if (n == 0) {
			throw std::runtime_error("EOF while reading");
		}
		already_read += n;
	}
}

void Channel::send_raw(uint8_t* data, size_t size) {
	size_t already_send = 0;
	while (already_send < size) {
		size_t n = write(&data[already_send], size - already_send);
		if (n == 0) {
			throw std::runtime_error("EOF while writing");
		}
		already_send += n;
	}
}


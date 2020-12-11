
#pragma once

#include <nlohmann/json.hpp>
#include <coro/Stream.h>

struct Channel {
	Channel(const std::string& fd_path);
	~Channel();

	Channel(Channel&& other);
	Channel& operator=(Channel&& other);

	nlohmann::json receive();
	void send(nlohmann::json response);

	void receive_raw(uint8_t* data, size_t size);
	void send_raw(uint8_t* data, size_t size);

	void close();

#ifdef __linux__
	int fd = -1;
#endif

#ifdef WIN32
	std::vector<uint8_t> info_buf;
	coro::Stream<asio::windows::stream_handle> stream;
#endif

private:
	size_t read(uint8_t* data, size_t size);
	size_t write(uint8_t* data, size_t size);
};

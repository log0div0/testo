
#pragma once

#include <nlohmann/json.hpp>

#ifdef WIN32
#include "winapi.hpp"
#endif

struct Channel {
	Channel() = default;
	Channel(const std::string& fd_path);
	~Channel();

	Channel(Channel&& other);
	Channel& operator=(Channel&& other);

	nlohmann::json receive();
	void send(const nlohmann::json& response);

	size_t read(uint8_t* data, size_t size);
	size_t write(uint8_t* data, size_t size);

#ifdef __linux__
	int fd = -1;
#endif

#ifdef WIN32
	winapi::File file;
#endif
};

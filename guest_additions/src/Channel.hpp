
#pragma once

#include <nlohmann/json.hpp>

#ifdef WIN32
#include "winapi.hpp"

typedef struct _tagVirtioPortInfo {
	UINT                Id;
	BOOLEAN             OutVqFull;
	BOOLEAN             HostConnected;
	BOOLEAN             GuestConnected;
	CHAR                Name[1];
}VIRTIO_PORT_INFO, * PVIRTIO_PORT_INFO;

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
	PVIRTIO_PORT_INFO getInfo();
	std::vector<uint8_t> info_buf;
	winapi::File file;
#endif
};

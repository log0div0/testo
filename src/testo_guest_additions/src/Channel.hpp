
#pragma once

#include <nlohmann/json.hpp>

struct Channel {
	nlohmann::json receive();
	void send(nlohmann::json response);

	void receive_raw(uint8_t* data, size_t size);
	void send_raw(uint8_t* data, size_t size);

	virtual size_t read(uint8_t* data, size_t size) = 0;
	virtual size_t write(uint8_t* data, size_t size) = 0;
};

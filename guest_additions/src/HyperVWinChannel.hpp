
#pragma once

#include <vector>
#include <cstdint>

struct HyperVWinChannel {
	HyperVWinChannel();
	~HyperVWinChannel();

	HyperVWinChannel(HyperVWinChannel&& other);
	HyperVWinChannel& operator=(HyperVWinChannel&& other);

	size_t read(uint8_t* data, size_t size);
	size_t write(uint8_t* data, size_t size);
	void close();
};

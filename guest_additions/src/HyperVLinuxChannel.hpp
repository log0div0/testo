
#pragma once

#include <vector>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

struct HyperVLinuxChannel {
	HyperVLinuxChannel();
	~HyperVLinuxChannel();

	HyperVLinuxChannel(HyperVLinuxChannel&& other);
	HyperVLinuxChannel& operator=(HyperVLinuxChannel&& other);

	size_t read(uint8_t* data, size_t size);
	size_t write(uint8_t* data, size_t size);
	void close();
};

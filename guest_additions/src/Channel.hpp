
#pragma once

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>

struct Channel {
	virtual size_t read(uint8_t* data, size_t size) = 0;
	virtual size_t write(uint8_t* data, size_t size) = 0;
	virtual void close() = 0;
};

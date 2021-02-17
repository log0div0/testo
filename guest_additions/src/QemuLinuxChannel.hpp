
#pragma once

#include <vector>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

struct QemuLinuxChannel {
	QemuLinuxChannel();
	~QemuLinuxChannel();

	QemuLinuxChannel(QemuLinuxChannel&& other);
	QemuLinuxChannel& operator=(QemuLinuxChannel&& other);

	size_t read(uint8_t* data, size_t size);
	size_t write(uint8_t* data, size_t size);
	void close();

	int fd = -1;
};

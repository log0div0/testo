
#pragma once

#include "Channel.hpp"

struct QemuLinuxChannel: Channel {
	QemuLinuxChannel();
	~QemuLinuxChannel();

	QemuLinuxChannel(QemuLinuxChannel&& other);
	QemuLinuxChannel& operator=(QemuLinuxChannel&& other);

	size_t read(uint8_t* data, size_t size) override;
	size_t write(uint8_t* data, size_t size) override;

	int fd = -1;
};

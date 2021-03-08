
#pragma once

#include "Channel.hpp"
#include <coro/Stream.h>

struct QemuWinChannel: Channel {
	QemuWinChannel();
	~QemuWinChannel();

	QemuWinChannel(QemuWinChannel&& other);
	QemuWinChannel& operator=(QemuWinChannel&& other);

	size_t read(uint8_t* data, size_t size) override;
	size_t write(uint8_t* data, size_t size) override;
	void close() override;

	std::vector<uint8_t> info_buf;
	coro::Stream<asio::windows::stream_handle> stream;
};

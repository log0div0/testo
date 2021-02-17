
#pragma once

#pragma once

#include <coro/Stream.h>

struct QemuWinChannel {
	QemuWinChannel();
	~QemuWinChannel();

	QemuWinChannel(QemuWinChannel&& other);
	QemuWinChannel& operator=(QemuWinChannel&& other);

	size_t read(uint8_t* data, size_t size);
	size_t write(uint8_t* data, size_t size);
	void close();

	std::vector<uint8_t> info_buf;
	coro::Stream<asio::windows::stream_handle> stream;
};

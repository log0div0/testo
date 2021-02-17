
#include "QemuWinChannel.hpp"
#include <coro/IoService.h>
#include <winapi/Functions.hpp>
#include <stdexcept>
#include "QemuWinChannelExtra.hpp"

QemuWinChannel::QemuWinChannel():
	stream(asio::windows::stream_handle(coro::IoService::current()->_impl))
{
	std::string device_path = GetVirtioDevicePath();
	HANDLE handle = CreateFile(winapi::utf8_to_utf16(device_path).c_str(),
		GENERIC_WRITE | GENERIC_READ,
		0,
		NULL,
		OPEN_EXISTING,
		FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
		NULL);
	if (handle == INVALID_HANDLE_VALUE) {
		throw std::runtime_error("CreateFile failed");
	}
	stream.handle().assign(handle);
	info_buf.resize(sizeof(VIRTIO_PORT_INFO));
}

QemuWinChannel::~QemuWinChannel() {
}

QemuWinChannel& QemuWinChannel::operator=(QemuWinChannel&& other) {
	std::swap(stream, other.stream);
	std::swap(info_buf, other.info_buf);
	return *this;
}

size_t QemuWinChannel::read(uint8_t* data, size_t size) {
	PVIRTIO_PORT_INFO info = GetVirtioDeviceInformation(stream.handle().native_handle(), info_buf);
	if (!info->HostConnected) {
		return 0;
	}
	return stream.readSome(asio::buffer(data, size));
}

size_t QemuWinChannel::write(uint8_t* data, size_t size) {
	return stream.write(asio::buffer(data, size));
}

void QemuWinChannel::close() {
	CloseHandle(stream.handle().native_handle());
}

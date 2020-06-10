
#include <Windows.h>
#include <vector>
#include "Channel_getInfo.hpp"
#include <stdexcept>

#define IOCTL_GET_INFORMATION    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

PVIRTIO_PORT_INFO Channel_getInfo(HANDLE handle, std::vector<uint8_t>& info_buf) {
	DWORD lpBytesReturned = 0;
	bool success = DeviceIoControl(
		handle,
		IOCTL_GET_INFORMATION,
		NULL,
		0,
		info_buf.data(),
		(DWORD)info_buf.size(),
		&lpBytesReturned,
		NULL);
	if (success) {
		return (PVIRTIO_PORT_INFO)info_buf.data();
	}
	if (GetLastError() != ERROR_MORE_DATA) {
		throw std::runtime_error("ERROR_MORE_DATA expected");
	}
	info_buf.resize(lpBytesReturned);
	success = DeviceIoControl(
		handle,
		IOCTL_GET_INFORMATION,
		NULL,
		0,
		info_buf.data(),
		(DWORD)info_buf.size(),
		&lpBytesReturned,
		NULL);
	if (!success) {
		throw std::runtime_error("DeviceIoControl failed");
	}
	return (PVIRTIO_PORT_INFO)info_buf.data();
}

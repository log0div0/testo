
#pragma once

#include <string>
#include <vector>
#include <Windows.h>

typedef struct _tagVirtioPortInfo {
	UINT                Id;
	BOOLEAN             OutVqFull;
	BOOLEAN             HostConnected;
	BOOLEAN             GuestConnected;
	CHAR                Name[1];
}VIRTIO_PORT_INFO, * PVIRTIO_PORT_INFO;

PVIRTIO_PORT_INFO GetVirtioDeviceInformation(HANDLE handle, std::vector<uint8_t>& info_buf);
std::string GetVirtioDevicePath();

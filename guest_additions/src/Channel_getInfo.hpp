
#pragma once

typedef struct _tagVirtioPortInfo {
	UINT                Id;
	BOOLEAN             OutVqFull;
	BOOLEAN             HostConnected;
	BOOLEAN             GuestConnected;
	CHAR                Name[1];
}VIRTIO_PORT_INFO, * PVIRTIO_PORT_INFO;

PVIRTIO_PORT_INFO Channel_getInfo(HANDLE handle, std::vector<uint8_t>& info_buf);


#include "QemuWinChannelExtra.hpp"
#include <stdexcept>
#include <winapi/Functions.hpp>

#include <setupapi.h>
#include <initguid.h>

#define IOCTL_GET_INFORMATION    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

PVIRTIO_PORT_INFO GetVirtioDeviceInformation(HANDLE handle, std::vector<uint8_t>& info_buf) {
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

DEFINE_GUID(GUID_VIOSERIAL_PORT,
0x6fde7521, 0x1b65, 0x48ae, 0xb6, 0x28, 0x80, 0xbe, 0x62, 0x1, 0x60, 0x26);
// {6FDE7521-1B65-48ae-B628-80BE62016026}

struct HardwareDeviceInfo {
	HardwareDeviceInfo(LPGUID interfaceGuid_): interfaceGuid(interfaceGuid_) {
		handle = SetupDiGetClassDevs(
			interfaceGuid,
			NULL,
			NULL,
			(DIGCF_PRESENT | DIGCF_DEVICEINTERFACE)
		);

		if (handle == INVALID_HANDLE_VALUE) {
			throw std::runtime_error("SetupDiGetClassDevs failed");
		}
	}

	~HardwareDeviceInfo() {
		if (handle) {
			SetupDiDestroyDeviceInfoList(handle);
			handle = NULL;
		}
	}

	HardwareDeviceInfo(HardwareDeviceInfo&& other);
	HardwareDeviceInfo& operator=(HardwareDeviceInfo&& other);

	std::string getDeviceInterfaceDetail() {
		SP_DEVICE_INTERFACE_DATA deviceInterfaceData = {};
		deviceInterfaceData.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);
		BOOL result = SetupDiEnumDeviceInterfaces(handle, 0, interfaceGuid, 0, &deviceInterfaceData);
		if (result == FALSE) {
			throw std::runtime_error("SetupDiEnumDeviceInterfaces failed");
		}

		ULONG requiredLength = 0;
		SetupDiGetDeviceInterfaceDetail(handle, &deviceInterfaceData, NULL, 0, &requiredLength, NULL);
		if (requiredLength == 0) {
			throw std::runtime_error("SetupDiGetDeviceInterfaceDetail failed (1)");
		}

		std::vector<uint8_t> buffer(requiredLength, 0);
		PSP_DEVICE_INTERFACE_DETAIL_DATA deviceInterfaceDetailData = (PSP_DEVICE_INTERFACE_DETAIL_DATA)buffer.data();
		deviceInterfaceDetailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

		ULONG length = requiredLength;
		result = SetupDiGetDeviceInterfaceDetail(
			handle,
			&deviceInterfaceData,
			deviceInterfaceDetailData,
			length,
			&requiredLength,
			NULL
		);
		if (result == FALSE) {
			throw std::runtime_error("SetupDiGetDeviceInterfaceDetail failed (2)");
		}

		if (deviceInterfaceDetailData->DevicePath == nullptr) {
			throw std::runtime_error("deviceInterfaceDetailData->DevicePath == nullptr");
		}

		return winapi::utf16_to_utf8(deviceInterfaceDetailData->DevicePath);
	}

	LPGUID interfaceGuid = NULL;
	HDEVINFO handle = NULL;
};

std::string GetVirtioDevicePath() {
	HardwareDeviceInfo info((LPGUID)&GUID_VIOSERIAL_PORT);
	std::string device_path = info.getDeviceInterfaceDetail();
	return device_path;
}

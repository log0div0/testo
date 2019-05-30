
#include "VirtualDisk.hpp"
#include <comdef.h>
#include <virtdisk.h>

VirtualDisk::VirtualDisk(const std::string& path) {
	VIRTUAL_STORAGE_TYPE storageType = {};

	OPEN_VIRTUAL_DISK_PARAMETERS parameters = {};
	parameters.Version = OPEN_VIRTUAL_DISK_VERSION_1;
	parameters.Version1.RWDepth = OPEN_VIRTUAL_DISK_RW_DEPTH_DEFAULT;

	auto result = OpenVirtualDisk(
		&storageType,
		bstr_t(path.c_str()),
		VIRTUAL_DISK_ACCESS_ALL,
		OPEN_VIRTUAL_DISK_FLAG_NONE,
		&parameters,
		&handle);

	if (result != ERROR_SUCCESS) {
		throw std::runtime_error("Failed to open virtual disk, error code = " + std::to_string(result));
	}
}

VirtualDisk::~VirtualDisk() {
	if (handle) {
		CloseHandle(handle);
		handle = INVALID_HANDLE_VALUE;
	}
}

void VirtualDisk::attach() {
	ATTACH_VIRTUAL_DISK_PARAMETERS parameters = {};
	parameters.Version = ATTACH_VIRTUAL_DISK_VERSION_1;

	auto result = AttachVirtualDisk(
		handle,
		nullptr,
		ATTACH_VIRTUAL_DISK_FLAG_PERMANENT_LIFETIME,
		0,
		&parameters,
		nullptr
	);

	if (result != ERROR_SUCCESS) {
		throw std::runtime_error("Failed to attach virtual disk, error code = " + std::to_string(result));
	}
}

void VirtualDisk::detach() {
	auto result = DetachVirtualDisk(
		handle,
		DETACH_VIRTUAL_DISK_FLAG_NONE,
		0
	);

	if (result != ERROR_SUCCESS) {
		throw std::runtime_error("Failed to detach virtual disk, error code = " + std::to_string(result));
	}
}

std::string VirtualDisk::physicalPath() const {
	wchar_t path[MAX_PATH];
	ULONG size = sizeof(path) * sizeof(wchar_t);
	auto result = GetVirtualDiskPhysicalPath(handle, &size, path);
	if (result != ERROR_SUCCESS) {
		throw std::runtime_error("Failed to get pthysical path of virtual disk, error code = " + std::to_string(result));
	}
	return (const char*)bstr_t(path);
}

bool VirtualDisk::isLoaded() const {
	GET_VIRTUAL_DISK_INFO info = {};
	info.Version = GET_VIRTUAL_DISK_INFO_IS_LOADED;
	ULONG size = sizeof(info);
	auto result = GetVirtualDiskInformation(handle, &size, &info, nullptr);
	if (result != ERROR_SUCCESS) {
		throw std::runtime_error("Failed to get info of virtual disk, error code = " + std::to_string(result));
	}
	return info.IsLoaded;
}

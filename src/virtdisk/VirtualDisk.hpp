
#pragma once

#include <windows.h>
#include <string>

struct VirtualDisk {
	VirtualDisk(const std::string& path);
	~VirtualDisk();

	VirtualDisk(const VirtualDisk& other) = delete;
	VirtualDisk& operator=(const VirtualDisk& other) = delete;

	void attach();
	void detach();

	std::string physicalPath() const;
	bool isLoaded() const;

private:
	HANDLE handle = INVALID_HANDLE_VALUE;
};

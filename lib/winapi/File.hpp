
#pragma once

#include <Windows.h>
#include <string>

namespace winapi {

struct File {
	File() = default;
	File(const std::string& path, DWORD dwDesiredAccess, DWORD dwCreationDisposition);
	~File();

	File(File&& other);
	File& operator=(File&& other);

	size_t read(uint8_t* data, size_t size);
	size_t write(const uint8_t* data, size_t size);
	uint64_t size() const;

	HANDLE handle = NULL;
};

}

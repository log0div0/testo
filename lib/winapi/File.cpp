
#include "File.hpp"
#include <stdexcept>
#include "Functions.hpp"

namespace winapi {

File::File(const std::string& path, DWORD dwDesiredAccess, DWORD dwCreationDisposition) {
	handle = CreateFile(winapi::utf8_to_utf16(path).c_str(),
		dwDesiredAccess,
		0,
		NULL,
		dwCreationDisposition,
		FILE_ATTRIBUTE_NORMAL,
		NULL);

	if (handle == INVALID_HANDLE_VALUE) {
		throw std::runtime_error("CreateFile failed");
	}
}

File::~File() {
	if (handle) {
		CloseHandle(handle);
	}
}

File::File(File&& other): handle(other.handle) {
	other.handle = NULL;
}

File& File::operator=(File&& other) {
	std::swap(handle, other.handle);
	return *this;
}

size_t File::read(uint8_t* data, size_t size) {
	DWORD result = 0;
	bool success = ReadFile(handle, data, (DWORD)size, &result, NULL);
	if (!success) {
		throw std::runtime_error("ReadFile failed");
	}
	return result;
}

size_t File::write(const uint8_t* data, size_t size) {
	DWORD result = 0;
	bool success = WriteFile(handle, data, (DWORD)size, &result, NULL);
	if (!success) {
		throw std::runtime_error("WriteFile failed");
	}
	return result;
}

uint64_t File::size() const {
	DWORD high = 0;
	DWORD low = GetFileSize(handle, &high);
	if (low == INVALID_FILE_SIZE) {
		throw std::runtime_error("GetFileSize failed");
	}
	return (uint64_t(high) << 32) | low;
}
}
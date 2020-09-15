
#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <experimental/filesystem>

namespace linuxapi {

struct File {
	File() = default;
	File(const std::experimental::filesystem::path& path, int flags, int mode): path(path) {
		handle = ::open(path.generic_string().c_str(), flags, mode);

		if (handle  < 0) {
			throw std::runtime_error("CreateFile " + path.generic_string() + " failed: " + strerror(errno));
		}
	}

	~File() {
		if (handle >= 0) {
			close(handle);
		}
	}

	File(File&& other): handle(other.handle) {
		other.handle = 0;
	}

	File& operator=(File&& other) {
		std::swap(handle, other.handle);
		return *this;
	}

	size_t read(uint8_t* data, size_t size) {
		auto result = ::read(handle, data, size);
		if (result < 0) {
			throw std::runtime_error("ReadFile " + path.generic_string() + " failed: " + strerror(errno));
		}
		return size_t(result);
	}

	size_t write(const uint8_t* data, size_t size) {
		auto result = ::write(handle, data, size);
		if (result < 0) {
			throw std::runtime_error("WriteFile " + path.generic_string() + " failed: " + strerror(errno));
		}
		return size_t(result);
	}

	std::experimental::filesystem::path path;
	int handle = -1;
};

}
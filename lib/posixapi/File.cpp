
#include "File.hpp"

#include <errno.h>
#include <stdexcept>
#include <string.h>

namespace posixapi {

File::File(const std::string& path, int flags, int mode) {
	handle = ::open(path.c_str(), flags, mode);

	if (handle  < 0) {
		throw std::runtime_error("CreateFile " + path + " failed: " + strerror(errno));
	}
}

File::~File() {
	if (handle >= 0) {
		close(handle);
	}
}

File::File(File&& other): handle(other.handle) {
	other.handle = 0;
}

File& File::operator=(File&& other) {
	std::swap(handle, other.handle);
	return *this;
}

size_t File::read(uint8_t* data, size_t size) {
	auto result = ::read(handle, data, size);
	if (result < 0) {
		throw std::runtime_error(std::string("::read failed: ") + strerror(errno));
	}
	return size_t(result);
}

size_t File::write(const uint8_t* data, size_t size) {
	auto result = ::write(handle, data, size);
	if (result < 0) {
		throw std::runtime_error(std::string("::write failed: ") + strerror(errno));
	}
	return size_t(result);
}

struct stat File::get_stat() {
	struct stat st = {};
	int error_code = ::fstat(handle, &st);
	if (error_code < 0) {
		throw std::runtime_error(std::string("::fstat failed: ") + strerror(errno));
	}
	return st;
}

}


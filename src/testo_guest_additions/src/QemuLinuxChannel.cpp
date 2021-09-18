
#include "QemuLinuxChannel.hpp"
#include <stdexcept>

#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

QemuLinuxChannel::QemuLinuxChannel() {
	std::string fd_path = "/dev/virtio-ports/negotiator.0";
	fd = open (fd_path.c_str(), O_RDWR);
	if (fd < 0) {
		std::string error_msg = "error " + std::to_string(errno) + " opening " + fd_path + ": " + strerror (errno);
		throw std::runtime_error(error_msg);
	}
}

QemuLinuxChannel::~QemuLinuxChannel() {
	if (fd >= 0) {
		::close(fd);
		fd = -1;
	}
}

QemuLinuxChannel& QemuLinuxChannel::operator=(QemuLinuxChannel&& other) {
	std::swap(fd, other.fd);
	return *this;
}

size_t QemuLinuxChannel::read(uint8_t* data, size_t size) {
	int n = ::read(fd, data, size);
	if (n < 0) {
		throw std::runtime_error("::read failed");
	}
	return n;
}

size_t QemuLinuxChannel::write(uint8_t* data, size_t size) {
	int n = ::write(fd, data, size);
	if (n < 0) {
		throw std::runtime_error("::write failed");
	}
	return n;
}

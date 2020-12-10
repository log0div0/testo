
#include "Stream.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>

#include <iostream>

namespace vir {

Stream::Stream(virStream* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Stream::~Stream() {
	if (handle) {
		virStreamFree(handle);
	}
}

Stream::Stream(Stream&& other): handle(other.handle) {
	other.handle = nullptr;
}

Stream& Stream::operator =(Stream&& other) {
	std::swap(handle, other.handle);
	return *this;
}

size_t Stream::recv_all(uint8_t* buf, size_t size) {
	uint32_t chunk_size = 2359312; //1024x768 ppm format
	size_t current_pos = 0;
	while (true) {
		if (size - current_pos < chunk_size) {
			throw std::runtime_error("receive: provided buffer size is not enough");
		}
		auto bytes_read = virStreamRecv(handle, (char*)(buf + current_pos), chunk_size); //1024x768 ppm format
		if (bytes_read < 0) {
			throw std::runtime_error(virGetLastErrorMessage());
		} else if (bytes_read == 0) {
			break;
		} else {
			//actually read something
			current_pos += bytes_read;
		}
	}

	return current_pos;
}

void Stream::finish() {
	if (virStreamFinish(handle)) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

}

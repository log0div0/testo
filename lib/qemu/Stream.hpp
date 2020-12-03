
#pragma once

#include <libvirt/libvirt.h>
#include <vector>
#include <stdint.h>

namespace vir {

struct Stream {
	Stream() = default;
	Stream(virStream* handle);
	~Stream();

	Stream(const Stream&) = delete;
	Stream& operator=(const Stream&) = delete;

	Stream(Stream&&);
	Stream& operator=(Stream&&);

	//Read until the end of stream, but it's not the equalent of recv_all from man
	size_t recv_all(uint8_t* buf, size_t size);
	void finish();

	::virStream* handle = nullptr;
};

}

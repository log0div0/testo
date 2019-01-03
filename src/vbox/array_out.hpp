
#pragma once

#include "api.hpp"

namespace vbox {

struct ArrayOut {
	ArrayOut() = default;
	~ArrayOut();

	ArrayOut(const ArrayOut&) = delete;
	ArrayOut& operator=(const ArrayOut&) = delete;
	ArrayOut(ArrayOut&& other);
	ArrayOut& operator=(ArrayOut&& other);

	uint8_t* data = nullptr;
	ULONG data_size = 0;
};

struct ArrayOutIface {
	ArrayOutIface() = default;
	~ArrayOutIface();

	ArrayOutIface(const ArrayOutIface&) = delete;
	ArrayOutIface& operator=(const ArrayOutIface&) = delete;
	ArrayOutIface(ArrayOutIface&& other);
	ArrayOutIface& operator=(ArrayOutIface&& other);

	IUnknown** ifaces = nullptr;
	ULONG ifaces_count = 0;
};

}

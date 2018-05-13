
#pragma once

#include "api.hpp"

namespace vbox {

struct ArrayOut {
	~ArrayOut();
	uint8_t* data = nullptr;
	ULONG data_size = 0;
};

struct ArrayOutIface {
	~ArrayOutIface();
	IUnknown** ifaces = nullptr;
	ULONG ifaces_count = 0;
};

}

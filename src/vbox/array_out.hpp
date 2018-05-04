
#pragma once

#include "api.hpp"

namespace vbox {

template <typename Value>
struct ArrayOut {
	~ArrayOut() {
		if (values) {
			api->pfnArrayOutFree(values);
		}
	}

	Value* values = nullptr;
	ULONG values_count = 0;
};

}

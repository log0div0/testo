
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

	Value operator[](size_t i) const {
		return values[i];
	}

	size_t size() const {
		return values_count;
	}

	Value* values = nullptr;
	ULONG values_count = 0;
};

}

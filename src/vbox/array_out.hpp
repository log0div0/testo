
#pragma once

#include <VBoxCAPIGlue/VBoxCAPIGlue.h>

namespace vbox {

template <typename Value>
struct ArrayOut {
	~ArrayOut() {
		if (values) {
			g_pVBoxFuncs->pfnArrayOutFree(values);
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

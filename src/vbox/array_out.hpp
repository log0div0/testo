
#pragma once

#include "api.hpp"

namespace vbox {

struct ArrayOut {
	~ArrayOut();
	void* values = nullptr;
	ULONG values_count = 0;
};

}

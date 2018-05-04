
#pragma once

#include "error.hpp"
#include "array_out.hpp"
#include <vector>

namespace vbox {

struct SafeArray {
	SafeArray();
	SafeArray(VARTYPE vt, size_t size);
	~SafeArray();

	SafeArray(const SafeArray&) = delete;
	SafeArray& operator=(const SafeArray&) = delete;

	SafeArray(SafeArray&& other);
	SafeArray& operator=(SafeArray&& other);

	SAFEARRAY* handle = nullptr;
};

}

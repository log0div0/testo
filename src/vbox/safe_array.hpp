
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

	void copy_in(void* data, size_t size);
	ArrayOut copy_out(VARTYPE vartype);
	ArrayOut copy_out();

	SAFEARRAY* handle = nullptr;
};

}

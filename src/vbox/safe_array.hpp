
#pragma once

#include "array_out.hpp"
#include <vector>

namespace vbox {

struct SafeArray {
	SafeArray();
	SafeArray(VARTYPE vt, ULONG size);
	~SafeArray();

	SafeArray(const SafeArray&) = delete;
	SafeArray& operator=(const SafeArray&) = delete;
	SafeArray(SafeArray&& other);
	SafeArray& operator=(SafeArray&& other);

	void copy_in(void* data, ULONG size);
	ArrayOut copy_out(VARTYPE vartype) const;
	ArrayOutIface copy_out_iface() const;

	int bitset() const;
	static SafeArray bitset(int bitset);

	SAFEARRAY* handle = nullptr;
};

}

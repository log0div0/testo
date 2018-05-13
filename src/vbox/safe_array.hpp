
#pragma once

#include "array_out.hpp"
#include <vector>

#ifdef WIN32
#define SAFEARRAY_AS_IN_PARAM(TYPE, safe_array) safe_array.handle
#define SAFEARRAY_AS_OUT_PARAM(TYPE, safe_array) &safe_array.handle
#else
#define SAFEARRAY_AS_IN_PARAM(TYPE, safe_array) safe_array.handle->c, (TYPE*)safe_array.handle->pv
#define SAFEARRAY_AS_OUT_PARAM(TYPE, safe_array) &safe_array.handle->c, (TYPE**)&safe_array.handle->pv
#endif

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

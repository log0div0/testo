
#pragma once

#include "array_out.hpp"
#include <vector>

#ifdef WIN32
#define SAFEARRAY_AS_IN_PARAM(TYPE, safe_array) safe_array.handle
#define SAFEARRAY_AS_OUT_PARAM(TYPE, safe_array) &safe_array.handle
#define SAFEARRAY_IN_PARAM(TYPE, param) SAFEARRAY* param
#define SAFEARRAY_OUT_PARAM(TYPE, param) SAFEARRAY** param
#define SAFEARRAY_FROM_IN_PARAM(TYPE, safe_array, param) safe_array = param
#define SAFEARRAY_TO_OUT_PARAM(TYPE, safe_array, param) *param = safe_array
#else
#define SAFEARRAY_AS_IN_PARAM(TYPE, safe_array) safe_array.handle->c, (TYPE*)safe_array.handle->pv
#define SAFEARRAY_AS_OUT_PARAM(TYPE, safe_array) &safe_array.handle->c, (TYPE**)&safe_array.handle->pv
#define SAFEARRAY_IN_PARAM(TYPE, param) ULONG param##_size, TYPE* param
#define SAFEARRAY_OUT_PARAM(TYPE, param) ULONG* param##_size, TYPE** param
#define SAFEARRAY_FROM_IN_PARAM(TYPE, safe_array, param) safe_array.handle->c = param##_size, safe_array.handle->pv = param
#define SAFEARRAY_TO_OUT_PARAM(TYPE, safe_array, param) *param##_size = safe_array.handle->c, *param = (TYPE*)safe_array.handle->pv, safe_array.handle->c = 0, safe_array.handle->pv = nullptr
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
	ArrayOut copy_out(VARTYPE vartype);
	ArrayOut copy_out_iface();

	SAFEARRAY* handle = nullptr;
};

}

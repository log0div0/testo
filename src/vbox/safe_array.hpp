
#pragma once

#include "array_out.hpp"
#include <vector>

#ifdef WIN32
#define ComSafeArrayIn(type, name) SAFEARRAY* name
#define ComSafeArrayOut(type, name) SAFEARRAY** name
#define ComSafeArrayInArg(f,t) (f)
#define ComSafeArrayOutArg(f,t) (&(f))
#else
#define ComSafeArrayIn(type, name) ULONG name##_size, type* name
#define ComSafeArrayOut(type, name) ULONG* name##_size, type** name
#define ComSafeArrayInArg(f,t) ((f) ? (f)->c : 0), ((f) ? (t*)((f)->pv) : NULL)
#define ComSafeArrayOutArg(f,t) (&((f)->c)), (t**)(&((f)->pv))
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
	ArrayOut copy_out();

	SAFEARRAY* handle = nullptr;
};

}

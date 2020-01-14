#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "quickjs/quickjs.h"
#pragma GCC diagnostic pop

#include <string>

namespace quickjs {

struct CString {
	CString() = delete;
	CString(const char* handle, JSContext* context);
	~CString();

	CString(const CString& other) = delete;
	CString& operator=(const CString& other) = delete;

	CString(CString&& other);
	CString& operator=(CString&& other);

	operator std::string();

	const char* handle;
	::JSContext* context = nullptr;
};


}
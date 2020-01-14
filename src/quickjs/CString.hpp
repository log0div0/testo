#pragma once
#include "quickjs/quickjs.h"

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

	::JSContext* context = nullptr;
	const char* handle;
};


}
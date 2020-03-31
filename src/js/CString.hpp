
#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include "quickjs/quickjs.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <string>

namespace js {

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

std::ostream& operator<<(std::ostream& stream, const CString& value);

}
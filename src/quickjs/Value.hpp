
#pragma once

#include "quickjs/quickjs.h"
#include "CString.hpp"

namespace quickjs {

struct Value {
	Value() = delete;
	Value(JSValue handle, JSContext* context);
	~Value();

	Value(const Value& other);
	Value& operator=(const Value& other);

	operator bool();
	operator CString();

	bool is_exception();
	bool is_error();
	bool is_undefined();
	bool is_bool();
	bool is_string();


	::JSContext* context = nullptr;
	::JSValue handle;
};

}

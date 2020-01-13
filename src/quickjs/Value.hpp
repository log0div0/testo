
#pragma once

#include "quickjs/quickjs.h"

namespace quickjs {

struct Value {
	Value() = delete;
	Value(JSValue handle, JSContext* context);
	~Value();

	Value(const Value& other);
	Value& operator=(const Value& other);

	operator bool();

	bool is_exception();
	bool is_bool();

	::JSContext* context = nullptr;
	::JSValue handle;
};

}


#pragma once

#include "quickjs/quickjs.h"

namespace quickjs {

struct Value {
	Value() = delete;
	Value(JSValue handle, JSContext* context);
	~Value();

	Value(const Value& other);
	Value& operator=(const Value& other);

	bool is_exception();

	::JSContext* context = nullptr;
	::JSValue handle;
};

}


#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include "quickjs/quickjs.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <memory>
#include <string>

namespace js {

struct Value;

struct ValueRef {
	ValueRef(JSValue handle, JSContext* context);

	operator std::string() const;

	bool is_exception() const;

	::JSValue handle;
	::JSContext* context = nullptr;
};

struct Value: ValueRef {
	Value() = delete;
	using ValueRef::ValueRef;
	~Value();

	Value(const Value& other);
	Value& operator=(const Value& other);
};

}

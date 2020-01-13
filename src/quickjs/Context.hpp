
#pragma once

#include "Value.hpp"
#include "quickjs/quickjs.h"
#include <string>

namespace quickjs {

struct Context {
	Context() = delete;
	Context(JSContext* handle);
	~Context();

	Context(const Context& other) = delete;
	Context& operator=(const Context& other) = delete;

	Context(Context&& other);
	Context& operator=(Context&& other);

	Value get_global_object();
	Value eval(const std::string& script);

	void register_global_function(const std::string& name, size_t length, JSValue (*f)(JSContext*, JSValueConst, int, JSValueConst*));

	Value create_bool(bool val);

	::JSContext* handle = nullptr;
};

}
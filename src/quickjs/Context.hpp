
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

	Value eval(const std::string& script);

	void set_opaque(void* opaque);

	Value get_global_object();
	void* get_opaque();
	Value get_exception();
	Value get_property_str(const Value& this_obj, const std::string& property);
	std::string get_last_error();

	void register_global_function(const std::string& name, size_t length, JSValue (*f)(JSContext*, JSValueConst, int, JSValueConst*));
	void register_nn_functions();

	Value create_bool(bool val);

	::JSContext* handle = nullptr;
};

}
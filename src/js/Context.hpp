
#pragma once

#include "Value.hpp"
#include <stb/Image.hpp>

namespace js {

struct ContextRef {
	ContextRef(JSContext* handle);

	Value eval(const std::string& script, bool compile_only = false);

	Value get_global_object();
	Value get_exception();

	Value throw_(Value val);

	Value new_bool(bool val);
	Value new_int32(int32_t val);
	Value new_undefined();
	Value new_string(const std::string& val);
	Value new_function(JSCFunction* f, const std::string& name, size_t length);
	Value new_array(size_t length);
	Value new_object();
	Value new_object_class(int class_id);

	void set_class_proto(JSClassID class_id, Value obj);

	::JSContext* handle = nullptr;

	stb::Image* image() const;

protected:
	void set_opaque(void* opaque);
	void* get_opaque() const;

	void register_global_function(const std::string& name, size_t length, JSCFunction* f);
};

struct Context: ContextRef {
	Context() = delete;
	Context(stb::Image* image);
	~Context();

	Context(const Context& other) = delete;
	Context& operator=(const Context& other) = delete;

	Context(Context&& other);
	Context& operator=(Context&& other);

private:
	void register_global_functions();
	void register_classes();
};

}
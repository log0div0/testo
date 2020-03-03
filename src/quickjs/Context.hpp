
#pragma once

#include "Value.hpp"

namespace quickjs {

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

static JSClassID nn_rect_class_id;

static JSClassDef nn_rect_class = {
	"NN_Rect",
	nullptr,
	nullptr,
	nullptr,
	nullptr,
};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

struct ContextRef {
	ContextRef(JSContext* handle);

	Value eval(const std::string& script, bool compile_only = false);

	void set_opaque(void* opaque);

	Value get_global_object();
	void* get_opaque();
	Value get_exception();
	std::string get_last_error();

	void register_global_function(const std::string& name, size_t length, JSCFunction* f);
	void register_nn_functions();

	Value throw_(Value val);

	Value new_bool(bool val);
	Value new_int32(int32_t val);
	Value new_undefined();
	Value new_string(const std::string& val);
	Value new_function(JSCFunction* f, const std::string& name, size_t length);
	Value new_array(size_t length);
	ObjectValue new_object_class(int class_id);

	void* mallocz(size_t size);
	void free(void* ptr);

	::JSContext* handle = nullptr;
};

struct Context: ContextRef {
	Context() = delete;
	using ContextRef::ContextRef;
	~Context();

	Context(const Context& other) = delete;
	Context& operator=(const Context& other) = delete;

	Context(Context&& other);
	Context& operator=(Context&& other);
};

}
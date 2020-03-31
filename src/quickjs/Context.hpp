
#pragma once

#include "Value.hpp"
#include <stb/Image.hpp>

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

	Value get_global_object();
	Value get_exception();
	std::string get_last_error();

	Value throw_(Value val);

	Value new_bool(bool val);
	Value new_int32(int32_t val);
	Value new_undefined();
	Value new_string(const std::string& val);
	Value new_function(JSCFunction* f, const std::string& name, size_t length);
	Value new_array(size_t length);
	Value new_object_class(int class_id);

	void* mallocz(size_t size);
	void free(void* ptr);

	::JSContext* handle = nullptr;

	stb::Image* image() const;

protected:
	void set_opaque(void* opaque);
	void* get_opaque() const;

	void register_global_function(const std::string& name, size_t length, JSCFunction* f);
	void register_global_functions();
};

struct Context: ContextRef {
	Context() = delete;
	Context(JSContext* handle, stb::Image* image);
	~Context();

	Context(const Context& other) = delete;
	Context& operator=(const Context& other) = delete;

	Context(Context&& other);
	Context& operator=(Context&& other);
};

}
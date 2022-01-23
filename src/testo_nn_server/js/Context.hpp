
#pragma once

#include "Value.hpp"
#include <stb/Image.hpp>
#include <vector>
#include <sstream>

namespace js {

struct ContextEnv {
	virtual ~ContextEnv() {}
	virtual stb::Image<stb::RGB> get_ref_image(const std::string& img_path) = 0;
};

struct ContextRef {
	struct Opaque {
		const stb::Image<stb::RGB>* image;
		std::stringstream _stdout;
		ContextEnv* env;
	};

	ContextRef(JSContext* handle);

	Value eval(const std::string& script, bool compile_only = false);
	Value call_constructor(Value constuctor, const std::vector<Value>& args);

	Value call(Value func, const ValueRef object, std::vector<Value>& args);

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
	Value new_continue_error(const std::string& message);

	void set_class_proto(JSClassID class_id, Value obj);

	::JSContext* handle = nullptr;

	const stb::Image<stb::RGB>* image() const;
	std::stringstream& get_stdout();
	ContextEnv* env() const;

protected:
	void set_opaque(void* opaque);
	void* get_opaque() const;


	void register_global_function(const std::string& name, size_t length, JSCFunction* f);
};

struct Context: ContextRef {
	Context() = delete;
	Context(const stb::Image<stb::RGB>* image, ContextEnv* env = nullptr);
	~Context();

	Context(const Context& other) = delete;
	Context& operator=(const Context& other) = delete;

	Context(Context&& other);
	Context& operator=(Context&& other);

private:
	Opaque opaque;
	void register_global_functions();
	void register_classes();
};

}

#include "Context.hpp"
#include "GlobalFunctions.hpp"
#include "FunctionsAdapters.hpp"
#include "Tensor.hpp"
#include <stdexcept>

namespace js {

ContextRef::ContextRef(::JSContext* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Value ContextRef::get_global_object() {
	Value result(JS_GetGlobalObject(handle), handle);
	if (result.is_exception()) {
		throw std::runtime_error("Error while getting global object");
	}
	return result;
}

void ContextRef::set_opaque(void* opaque) {
	JS_SetContextOpaque(handle, opaque);
}

void* ContextRef::get_opaque() const {
	return JS_GetContextOpaque(handle);
}

Value ContextRef::eval(const std::string& script, bool compile_only) {
	int flags = JS_EVAL_TYPE_GLOBAL;
	if (compile_only) {
		flags |= JS_EVAL_FLAG_COMPILE_ONLY;
	}

	Value result(JS_Eval(handle, script.c_str(), script.length(), "<input>", flags), handle);
	if (result.is_exception()) {
		throw std::runtime_error(get_last_error());
	}

	return result;
}

Value ContextRef::new_bool(bool val) {
	return Value(JS_NewBool(handle, val), handle);
}

Value ContextRef::new_int32(int32_t val) {
	return Value(JS_NewInt32(handle, val), handle);
}

Value ContextRef::new_undefined() {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
	return Value(JS_UNDEFINED, handle);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

Value ContextRef::new_string(const std::string& val) {
	return Value(JS_NewString(handle, val.c_str()), handle);
}

Value ContextRef::new_function(JSCFunction* f, const std::string& name, size_t length) {
	return Value(JS_NewCFunction(handle, f, name.c_str(), length), handle);
}

Value ContextRef::new_array(size_t length) {
	return Value(JS_NewArray(handle), handle);;
}

Value ContextRef::new_object() {
	return Value(JS_NewObject(handle), handle);
}

Value ContextRef::new_object_class(int class_id) {
	Value val(JS_NewObjectClass(handle, class_id), handle);
	if (val.is_exception()) {
		throw std::runtime_error("JS_NewObjectClass failed");
	}
	return val;
}

void ContextRef::set_class_proto(JSClassID class_id, Value obj) {
	JS_SetClassProto(handle, class_id, obj.release());
}

void* ContextRef::mallocz(size_t size) {
	return js_mallocz(handle, size);
}

void ContextRef::free(void* ptr) {
	js_free(handle, ptr);
}

Value ContextRef::throw_(Value val) {
	return Value(JS_Throw(handle, val.release()), handle);
}

void ContextRef::register_global_function(const std::string& name, size_t length, JSCFunction* f) {
	get_global_object().set_property(name, new_function(f, name, length));
}

Value ContextRef::get_exception() {
	return Value(JS_GetException(handle), handle);
}

std::string ContextRef::get_last_error() {
	std::string result;

	Value exception_val = get_exception();
	if (!exception_val.is_error()) {
		result += "Throw: ";
	}

	std::string exception_str(exception_val);
	result += exception_str;

	if (exception_val.is_error()) {
		Value val = exception_val.get_property("stack");
		if (!val.is_undefined()) {
			std::string stack(val);
			result += stack;
			result += "\n";
		}
	}

	return result;
}

stb::Image* ContextRef::image() const {
	if (!get_opaque()) {
		throw std::runtime_error("Context opaque is nullptr");
	}
	return (stb::Image*)get_opaque();
}

Context::Context(stb::Image* image): ContextRef(JS_NewContext(Runtime::instance().handle)) {
	// image может быть нулевым, если мы просто хотим скомпилировать js
	set_opaque(image);

	register_global_functions();
	register_classes();
}

Context::~Context() {
	if (handle) {
		JS_FreeContext(handle);
	}
}

Context::Context(Context&& other): ContextRef(other.handle) {
	other.handle = nullptr;
}

Context& Context::operator=(Context&& other) {
	std::swap(handle, other.handle);
	return *this;
}

void Context::register_global_functions() {
	register_global_function("print", 1, Func<print>);
	register_global_function("find_text", 1, Func<find_text>);
}

void Context::register_classes() {
	Tensor::register_class(*this);
}

}


#include "Context.hpp"
#include "GlobalFunctions.hpp"
#include "FunctionsAdapters.hpp"
#include "Tensor.hpp"
#include "Point.hpp"
#include <stdexcept>
#include <iostream>

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
		Value exception_val = get_exception();
		if (exception_val.is_instance_of(get_global_object().get_property_str("ContinueError"))) {
			std::string message = exception_val.get_property_str("message");
			throw nn::ContinueError(message);
		} else {
			std::string message;

			if (!exception_val.is_error()) {
				message += "Throw: ";
			}

			std::string exception_str(exception_val);
			message += exception_str;

			if (exception_val.is_error()) {
				Value val = exception_val.get_property_str("stack");
				if (!val.is_undefined()) {
					std::string stack(val);
					message += stack;
					message += "\n";
				}
			}

			throw std::runtime_error(message);
		}
	}

	return result;
}

Value ContextRef::call_constructor(Value constuctor, const std::vector<Value>& args) {
	std::vector<JSValueConst> argv;
	for (auto& arg: args) {
		argv.push_back(arg.handle);
	}
	JSValue result = JS_CallConstructor(handle, constuctor.handle, argv.size(), argv.data());
	return {result, handle};
}

Value ContextRef::call(Value func, const ValueRef object, std::vector<Value>& args) {

	std::vector<JSValueConst> argv;
	for (auto& arg: args) {
		argv.push_back(arg.handle);
	}

	JSValue result = JS_Call(handle, func.handle, object.handle, argv.size(), argv.data());
	return {result, handle};
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

Value ContextRef::new_continue_error(const std::string& message) {
	return call_constructor(get_global_object().get_property_str("ContinueError"), {new_string(message)});
}

void ContextRef::set_class_proto(JSClassID class_id, Value obj) {
	JS_SetClassProto(handle, class_id, obj.release());
}

Value ContextRef::throw_(Value val) {
	return Value(JS_Throw(handle, val.release()), handle);
}

void ContextRef::register_global_function(const std::string& name, size_t length, JSCFunction* f) {
	get_global_object().set_property_str(name, new_function(f, name, length));
}

Value ContextRef::get_exception() {
	return Value(JS_GetException(handle), handle);
}

const stb::Image<stb::RGB>* ContextRef::image() const {
	if (!get_opaque()) {
		throw std::runtime_error("Context opaque is nullptr");
	}
	return ((Opaque*)get_opaque())->image;
}

std::stringstream& ContextRef::stdout() {
	if (!get_opaque()) {
		throw std::runtime_error("Context opaque is nullptr");
	}
	return ((Opaque*)get_opaque())->stdout;
}

std::shared_ptr<Channel> ContextRef::channel() const {
	if (!get_opaque()) {
		throw std::runtime_error("Context opaque is nullptr");
	}
	return ((Opaque*)get_opaque())->channel;
}

Context::Context(const stb::Image<stb::RGB>* image, std::shared_ptr<Channel> channel): ContextRef(JS_NewContext(Runtime::instance().handle)) {
	// image может быть нулевым, если мы просто хотим скомпилировать js
	opaque.image = image;
	opaque.channel = channel;
	set_opaque((void*)&opaque);

	register_global_functions();
	register_classes();

	eval(R"(
		function ContinueError(message) {
			if (!new.target) {
				return new ContinueError(message);
			}
			this.name = "ContinueError";
			this.message = message;
		}

		ContinueError.prototype = Object.create(Error.prototype);
	)");
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
	register_global_function("find_img", 1, Func<find_img>);
}

void Context::register_classes() {
	TextTensor::register_class(*this);
	ImgTensor::register_class(*this);
	Point::register_class(*this);
}

}

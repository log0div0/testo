
#include "Context.hpp"
#include "nn/Context.hpp"
#include <stdexcept>
#include <iostream>

namespace quickjs {



Value js_print(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	for (size_t i = 0; i < args.size(); i++) {
		if (i != 0) {
			std::cout << ' ';
		}
		std::cout << args[i];
	}
	std::cout << std::endl;
	return ctx.new_undefined();
}


Value detect_text(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	if (args.size() > 3) {
		throw std::runtime_error("Invalid arguments count in detect_text");
	}

	nn::Context* nn_context = (nn::Context*)ctx.get_opaque();

	std::string text, color, background_color;
	text = std::string(args.at(0));

	if (args.size() > 1) {
		color = std::string(args.at(1));
	}

	if (args.size() > 2) {
		background_color = std::string(args.at(2));
	}

	auto result = nn_context->ocr().search(text, color, background_color);
	return ctx.new_bool(result.size());
}



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

void* ContextRef::get_opaque() {
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

Value ContextRef::throw_(Value val) {
	return Value(JS_Throw(handle, val.release()), handle);
}

void ContextRef::register_global_function(const std::string& name, size_t length, JSCFunction* f) {
	get_global_object().set_property_str(name, new_function(f, name, length));
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
		Value val = exception_val.get_property_str("stack");
		if (!val.is_undefined()) {
			std::string stack(val);
			result += stack;
			result += "\n";
		}
	}

	return result;
}

typedef Value FunctionType(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args);

template <FunctionType F>
JSValue js_func(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
	ContextRef ctx_ref(ctx);
	try {
		std::vector<ValueRef> args;
		for (int i = 0; i < argc; ++i) {
			args.push_back(ValueRef(argv[i], ctx));
		}
		Value result = F(ctx, ValueRef(this_val, ctx), args);
		return result.release();
	} catch (const std::exception& error) {
		Value exception = ctx_ref.throw_(ctx_ref.new_string(error.what()));
		return exception.release();
	}
}

void ContextRef::register_nn_functions() {
	register_global_function("print", 1, js_func<js_print>);
	register_global_function("detect_text", 1, js_func<detect_text>);
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

}

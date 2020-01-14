
#include "Context.hpp"
#include "nn/text_detector/TextDetector.hpp"
#include <stdexcept>
#include <iostream>

namespace quickjs {

Context::Context(::JSContext* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Context::~Context() {
	if (handle) {
		JS_FreeContext(handle);
	}
}

Context::Context(Context&& other): handle(other.handle) {
	other.handle = nullptr;
}

Context& Context::operator=(Context&& other) {
	std::swap(handle, other.handle);
}

Value Context::get_global_object() {
	Value result(JS_GetGlobalObject(handle), handle);
	if (result.is_exception()) {
		throw std::runtime_error("Error while getting global object");
	}
	return result;
}

void Context::set_opaque(void* opaque) {
	JS_SetContextOpaque(handle, opaque);
}

void* Context::get_opaque() {
	return JS_GetContextOpaque(handle);
}

Value Context::eval(const std::string& script) {
	Value result(JS_Eval(handle, script.c_str(), script.length(), "<input>", JS_EVAL_TYPE_GLOBAL), handle);
	if (result.is_exception()) {
		throw std::runtime_error(get_last_error());
	}

	return result;
}

Value Context::create_bool(bool val) {
	return Value(JS_NewBool(handle, true), handle);
}

void Context::register_global_function(const std::string& name, size_t length, JSValue (*f)(JSContext*, JSValueConst, int, JSValueConst*)) {
	auto global = get_global_object();
	if (JS_SetPropertyStr(handle, global.handle, name.c_str(), JS_NewCFunction(handle, f, name.c_str(), length)) < 0) {
		throw std::runtime_error("Can't register global function " + name);
	}
}

Value Context::get_exception() {
	return Value(JS_GetException(handle), handle);
}

Value Context::get_property_str(const Value& this_obj, const std::string& property) {
	return Value(JS_GetPropertyStr(handle, this_obj.handle, property.c_str()), handle);
}

//TODO: set_property_str!

std::string Context::get_last_error() {
	std::string result;

	Value exception_val = get_exception();
	if (!exception_val.is_error()) {
		result += "Throw: ";
	}

	CString exception_str(exception_val);
	result += exception_str;

	if (exception_val.is_error()) {
		Value val = get_property_str(exception_val, "stack");
		if (!val.is_undefined()) {
			CString stack(val);
			result += stack;
			result += "\n";
		}
	}

	return result;
}

JSValue detect_text(JSContext* ctx, JSValueConst this_val, int argc, JSValueConst* argv) {
	if (argc > 3) {
		return JS_EXCEPTION;
	}

	stb::Image* current_image = (stb::Image*)JS_GetContextOpaque(ctx);

	const char* text = nullptr;
	const char* foreground = nullptr;
	const char* background = nullptr;

	std::string text_string, foreground_string, background_string;
	text = JS_ToCString(ctx, argv[0]);
	text_string = text;

	if (argc > 1) {
		foreground = JS_ToCString(ctx, argv[1]);
		foreground_string = foreground;
	}

	if (argc > 2) {
		background = JS_ToCString(ctx, argv[2]);
		background_string = background;
	}

	auto result = TextDetector::instance().detect(*current_image, text_string, foreground_string, background_string);

	if (text) {
		JS_FreeCString(ctx, text);
	}

	if (foreground) {
		JS_FreeCString(ctx, foreground);
	}

	if (background) {
		JS_FreeCString(ctx, background);
	}

	auto res = JS_NewBool(ctx, result.size());
	return res;
}

JSValue js_print(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
	int i;
	const char *str;

	for(i = 0; i < argc; i++) {
		if (i != 0) {
			putchar(' ');
		}
		str = JS_ToCString(ctx, argv[i]);
		if (!str) {
			return JS_EXCEPTION;
		}
		fputs(str, stdout);
		JS_FreeCString(ctx, str);
	}
	putchar('\n');
	return JS_UNDEFINED;
}

void Context::register_nn_functions() {
	register_global_function("detect_text", 1, detect_text);
	register_global_function("print", 1, js_print);
}

}


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
		throw std::runtime_error("Error while executing javascript");
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

JSValue detect_text(JSContext* ctx, JSValueConst this_val, int argc, JSValueConst* argv) {
	stb::Image* current_image = (stb::Image*)JS_GetContextOpaque(ctx);

	const char* text;
	text = JS_ToCString(ctx, argv[0]);
	auto result = TextDetector::instance().detect(*current_image, text, "", "");
	JS_FreeCString(ctx, text);
	auto res = JS_NewBool(ctx, result.size());
	return res;
}

void Context::register_nn_functions() {
	register_global_function("detect_text", 1, detect_text);
}

}

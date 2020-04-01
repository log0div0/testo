
#pragma once

#include "Runtime.hpp"
#include <vector>

namespace js {

typedef JSValue JSCGetter(JSContext *ctx, JSValueConst this_val);
typedef JSValue JSCSetter(JSContext *ctx, JSValueConst this_val, JSValueConst val);

typedef Value JSCPPFunction(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args);
typedef Value JSCPPGetter(ContextRef ctx, ValueRef this_val);
typedef Value JSCPPSetter(ContextRef ctx, ValueRef this_val, ValueRef val);

template <JSCPPFunction F>
JSValue Func(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
	try {
		std::vector<ValueRef> args;
		for (int i = 0; i < argc; ++i) {
			args.push_back(ValueRef(argv[i], ctx));
		}
		Value result = F(ctx, ValueRef(this_val, ctx), args);
		return result.release();
	} catch (const std::exception& error) {
		ContextRef ctx_ref(ctx);
		Value exception = ctx_ref.throw_(ctx_ref.new_string(error.what()));
		return exception.release();
	}
}

template <JSCPPGetter F>
JSValue Func(JSContext* ctx, JSValueConst this_val) {
	try {
		Value result = F(ctx, ValueRef(this_val, ctx));
		return result.release();
	} catch (const std::exception& error) {
		ContextRef ctx_ref(ctx);
		Value exception = ctx_ref.throw_(ctx_ref.new_string(error.what()));
		return exception.release();
	}
}

template <JSCPPSetter F>
JSValue Func(JSContext* ctx, JSValueConst this_val, JSValueConst val) {
	try {
		Value result = F(ctx, ValueRef(this_val, ctx), ValueRef(val, ctx));
		return result.release();
	} catch (const std::exception& error) {
		ContextRef ctx_ref(ctx);
		Value exception = ctx_ref.throw_(ctx_ref.new_string(error.what()));
		return exception.release();
	}
}

template <JSCPPFunction func>
JSCFunctionListEntry Method(const char* name, uint8_t length) {
	JSCFunctionListEntry entry = {};
	entry.name = name;
	entry.prop_flags = JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE;
	entry.def_type = JS_DEF_CFUNC;
	entry.u.func.length = length;
	entry.u.func.cproto = JS_CFUNC_generic;
	entry.u.func.cfunc.generic = Func<func>;
	return entry;
}

template <JSCPPGetter getter, JSCPPSetter setter>
JSCFunctionListEntry GetSet(const char* name) {
	JSCFunctionListEntry entry = {};
	entry.name = name;
	entry.prop_flags = JS_PROP_CONFIGURABLE;
	entry.def_type = JS_DEF_CGETSET;
	entry.u.getset.get.getter = Func<getter>;
	entry.u.getset.set.setter = Func<setter>;
	return entry;
}

inline JSCFunctionListEntry Prop(const char* name, const char* value, uint8_t prop_flags = 0) {
	JSCFunctionListEntry entry = {};
	entry.name = name;
	entry.prop_flags = prop_flags;
	entry.def_type = JS_DEF_PROP_STRING;
	entry.u.str = value;
	return entry;
}

}

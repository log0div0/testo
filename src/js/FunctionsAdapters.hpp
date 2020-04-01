
#pragma once

#include "Runtime.hpp"
#include <vector>

namespace js {

typedef Value CFunctionType(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args);

template <CFunctionType F>
JSValue CFunction(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
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

}


#pragma once

#include "Value.hpp"
#include <vector>
#include <sstream>

namespace js {

struct ContextRef {
	ContextRef(JSContext* handle);

	void compile(const std::string& script);

	Value get_exception();

	::JSContext* handle = nullptr;
};

struct Context: ContextRef {
	Context();
	~Context();

	Context(const Context& other) = delete;
	Context& operator=(const Context& other) = delete;

	Context(Context&& other);
	Context& operator=(Context&& other);
};

}

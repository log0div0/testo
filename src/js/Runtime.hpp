
#pragma once

#include "Context.hpp"

namespace js {

struct RuntimeRef {
	RuntimeRef(::JSRuntime* handle);

	Context create_context(stb::Image* image);

	::JSRuntime* handle = nullptr;
};

struct Runtime: RuntimeRef {
	Runtime() = default;
	Runtime(::JSRuntime* handle);
	~Runtime();

	Runtime(const Runtime&) = delete;
	Runtime& operator=(const Runtime&) = delete;

	Runtime(Runtime&& other);
	Runtime& operator=(Runtime&&);

};

Runtime create_runtime();

}

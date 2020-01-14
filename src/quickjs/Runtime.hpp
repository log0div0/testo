#pragma once

#include "Context.hpp"

namespace quickjs {

struct Runtime {
	Runtime() = default;
	Runtime(::JSRuntime* handle);
	~Runtime();

	Runtime(const Runtime&) = delete;
	Runtime& operator=(const Runtime&) = delete;

	Runtime(Runtime&& other);
	Runtime& operator=(Runtime&&);

	Context create_context();

	::JSRuntime* handle = nullptr;
};

Runtime create_runtime();

}
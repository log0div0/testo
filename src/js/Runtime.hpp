
#pragma once

#include "Context.hpp"

namespace js {

struct RuntimeRef {
	RuntimeRef(::JSRuntime* handle);

	::JSRuntime* handle = nullptr;
};

struct Runtime: RuntimeRef {
	static Runtime& instance();

	Runtime(const Runtime&) = delete;
	Runtime& operator=(const Runtime&) = delete;

	Runtime(Runtime&& other) = delete;
	Runtime& operator=(Runtime&&) = delete;

private:
	Runtime();
	~Runtime();
};

}

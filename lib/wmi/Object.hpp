
#pragma once

#include "Error.hpp"

namespace wmi {

template <typename IObject>
struct Object {
	Object() {}
	Object(IObject* handle_): handle(handle_) {
		try {
			if (handle == nullptr) {
				throw std::runtime_error("nullptr");
			}
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
	~Object() {
		if (handle) {
			handle->Release();
			handle = nullptr;
		}
	}

	Object(const Object& other): handle(other.handle) {
		handle->AddRef();
	}
	Object& operator=(const Object& other) {
		if (handle) {
			handle->Release();
			handle = nullptr;
		}
		handle = other.handle;
		handle->AddRef();
		return *this;
	}
	Object(Object&& other): handle(other.handle) {
		other.handle = nullptr;
	}
	Object& operator=(Object&& other) {
		std::swap(handle, other.handle);
		return *this;
	}
	IObject* handle = nullptr;
};

}


#include "medium.hpp"
#include <stdexcept>
#include <ostream>
#include "error.hpp"
#include "string.hpp"

namespace vbox {

Medium::Medium(IMedium* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Medium::~Medium() {
	if (handle) {
		IMedium_Release(handle);
	}
}

Medium::Medium(Medium&& other): handle(other.handle) {
	other.handle = nullptr;
}

Medium& Medium::operator=(Medium&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Medium::name() const {
	try {
		BSTR name = nullptr;
		HRESULT rc = IMedium_get_Name(handle, &name);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Medium::operator bool() const {
	return handle != nullptr;
}

std::ostream& operator<<(std::ostream& stream, const Medium& medium) {
	stream << medium.name();
	return stream;
}

}

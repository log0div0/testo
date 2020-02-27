#include "CString.hpp"
#include <stdexcept>
#include <iostream>

namespace quickjs {

CString::CString(const char* handle, JSContext* context): handle(handle), context(context) {
	if (!handle || !context) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

CString::~CString() {
	if (context) {
		JS_FreeCString(context, handle);
	}
}

CString::CString(CString&& other): handle(other.handle), context(other.context) {
	other.handle = nullptr;
	other.context = nullptr;
}

CString& CString::operator=(CString&& other) {
	std::swap(handle, other.handle);
	std::swap(context, other.context);
	return *this;
}

CString::operator std::string() {
	return std::string(handle);
}

std::ostream& operator<<(std::ostream& stream, const CString& value) {
	return stream << value.handle;
}

}
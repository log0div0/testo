#include "CString.hpp"
#include <stdexcept>

namespace quickjs {

CString::CString(const char* handle, JSContext* context): handle(handle), context(context) {
	if (!handle || !context) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

CString::~CString() {
	JS_FreeCString(context, handle);
}

CString::CString(CString&& other): handle(other.handle), context(other.context) {
	other.handle = nullptr;
	other.context = nullptr;
}

CString& CString::operator=(CString&& other) {
	std::swap(handle, other.handle);
	std::swap(context, other.context);
}

CString::operator std::string() {
	return std::string(handle);
}

}
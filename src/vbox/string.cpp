
#include "string.hpp"
#include "error.hpp"
#include <ostream>

namespace vbox {

BaseUtf8String::operator std::string() const {
	return data;
}

BaseUtf8String::operator std::wstring() const {
	return Utf16String(*this);
}

BaseUtf16String::operator std::string() const {
	return Utf8String(*this);
}

BaseUtf16String::operator std::wstring() const {
	return (wchar_t*)data;
}

BaseUtf16String::BaseUtf16String(BSTR data): data(data) {}

BaseUtf16String::BaseUtf16String(BaseUtf16String&& other): data(other.data) {
	other.data = nullptr;
}

Utf8String::Utf8String(const BaseUtf16String& other) {
	try {
		HRESULT rc = api->pfnUtf16ToUtf8(other.data, &data);
		if (FAILED(rc)) {
			throw Error(rc);
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Utf8String::~Utf8String() {
	if (data) {
		api->pfnUtf8Free(data);
	}
}

std::ostream& operator<<(std::ostream& stream, const Utf8String& string) {
	return stream << string.data;
}

Utf16String::Utf16String(const BaseUtf8String& other) {
	try {
		HRESULT rc = api->pfnUtf8ToUtf16(other.data, &data);
		if (FAILED(rc)) {
			throw Error(rc);
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Utf16String::~Utf16String() {
	if (data) {
		api->pfnUtf16Free(data);
	}
}

String::~String() {
	if (data) {
		api->pfnComUnallocString(data);
	}
}

String::String(BSTR data): BaseUtf16String(data) {
	if (!data) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

}

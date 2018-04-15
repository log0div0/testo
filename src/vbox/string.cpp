
#include "string.hpp"
#include "error.hpp"

namespace vbox {

BaseUtf8String::operator std::string() const {
	return data;
}

BaseUtf16String::operator std::string() const {
	return Utf8String(*this);
}

BaseUtf8String::operator std::wstring() const {
	return Utf16String(*this);
}

BaseUtf16String::operator std::wstring() const {
	return (wchar_t*)data;
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

COMString::~COMString() {
	if (data) {
		api->pfnComUnallocString(data);
	}
}

}

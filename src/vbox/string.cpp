
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

Utf8String::Utf8String(const BaseUtf16String& utf16) {
	try {
		HRESULT rc = g_pVBoxFuncs->pfnUtf16ToUtf8(utf16.data, &data);
		if (FAILED(rc)) {
			throw Error(rc);
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Utf8String::~Utf8String() {
	if (data) {
		g_pVBoxFuncs->pfnUtf8Free(data);
	}
}

Utf16String::Utf16String(const BaseUtf8String& utf8) {
	try {
		HRESULT rc = g_pVBoxFuncs->pfnUtf8ToUtf16(utf8.data, &data);
		if (FAILED(rc)) {
			throw Error(rc);
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Utf16String::~Utf16String() {
	if (data) {
		g_pVBoxFuncs->pfnUtf16Free(data);
	}
}

COMString::~COMString() {
	if (data) {
		g_pVBoxFuncs->pfnComUnallocString(data);
	}
}

}

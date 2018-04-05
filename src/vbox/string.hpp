
#pragma once

#include <VBoxCAPIGlue/VBoxCAPIGlue.h>
#include <string>

namespace vbox {

struct Utf8String;
struct Utf16String;

struct BaseUtf8String {
	char* data = nullptr;
	operator std::string() const;
	operator std::wstring() const;
};

struct BaseUtf16String {
	BSTR data = nullptr;
	operator std::string() const;
	operator std::wstring() const;
};

struct Utf8String: BaseUtf8String {
	Utf8String() = default;
	Utf8String(const BaseUtf16String& utf16);
	~Utf8String();
};

struct Utf16String: BaseUtf16String {
	Utf16String() = default;
	Utf16String(const BaseUtf8String& utf8);
	~Utf16String();
};

struct COMString: BaseUtf16String {
	COMString() = default;
	~COMString();
};

}

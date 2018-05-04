
#pragma once

#include "api.hpp"
#include <string>

namespace vbox {

struct Utf8String;
struct Utf16String;

struct BaseUtf8String {
	char* data = nullptr;
	operator std::string() const;
	operator std::wstring() const;

	BaseUtf8String() = default;

	BaseUtf8String(const BaseUtf8String&) = delete;
	BaseUtf8String& operator=(const BaseUtf8String&) = delete;
	BaseUtf8String(BaseUtf8String&&);
	BaseUtf8String& operator=(BaseUtf8String&&);

protected:
	BaseUtf8String(char* data);
};

struct BaseUtf16String {
	BSTR data = nullptr;
	operator std::string() const;
	operator std::wstring() const;

	BaseUtf16String() = default;

	BaseUtf16String(const BaseUtf16String&) = delete;
	BaseUtf16String& operator=(const BaseUtf16String&) = delete;
	BaseUtf16String(BaseUtf16String&&);
	BaseUtf16String& operator=(BaseUtf16String&&);

protected:
	BaseUtf16String(BSTR data);
};

struct Utf8String: BaseUtf8String {
	Utf8String() = default;
	Utf8String(const BaseUtf16String& utf16);
	~Utf8String();
};

std::ostream& operator<<(std::ostream& stream, const Utf8String&);

struct Utf16String: BaseUtf16String {
	Utf16String() = default;
	Utf16String(const BaseUtf8String& utf8);
	~Utf16String();
};

std::wostream& operator<<(std::wostream& stream, const Utf16String&);

struct String: BaseUtf16String {
	String() = default;
	String(BSTR data);
	~String();

	String(String&&) = default;
	String& operator=(String&&) = default;
};

}

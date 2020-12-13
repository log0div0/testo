
#include "UTF.hpp"
#include <stdexcept>

namespace winapi {

std::wstring utf8_to_utf16(const std::string& utf8) {
	if (utf8.empty()) {
		return {};
	}
	int num_chars = MultiByteToWideChar(
		CP_UTF8,
		MB_ERR_INVALID_CHARS,
		utf8.c_str(),
		(int)utf8.length(),
		NULL,
		0
	);
	if (num_chars == 0) {
		throw std::runtime_error("MultiByteToWideChar failed (1)");
	}
	std::wstring utf16;
	utf16.resize(num_chars);
	int num_chars2 = MultiByteToWideChar(
		CP_UTF8,
		MB_ERR_INVALID_CHARS,
		utf8.c_str(),
		(int)utf8.length(),
		&utf16[0],
		num_chars
	);
	if (num_chars2 == 0) {
		throw std::runtime_error("MultiByteToWideChar failed (2)");
	}
	return utf16;
}

std::string utf16_to_utf8(const std::wstring& utf16) {
	if (utf16.empty()) {
		return {};
	}
	int num_bytes = WideCharToMultiByte(
		CP_UTF8,
		WC_ERR_INVALID_CHARS,
		utf16.c_str(),
		(int)utf16.length(),
		NULL,
		0,
		NULL,
		NULL);
	if (num_bytes == 0) {
		throw std::runtime_error("WideCharToMultiByte failed (1)");
	}
	std::string utf8;
	utf8.resize(num_bytes);
	int num_bytes2 = WideCharToMultiByte(
		CP_UTF8,
		WC_ERR_INVALID_CHARS,
		utf16.c_str(),
		(int)utf16.length(),
		&utf8[0],
		num_bytes,
		NULL,
		NULL);
	if (num_bytes2 == 0) {
		throw std::runtime_error("WideCharToMultiByte failed (2)");
	}
	return utf8;
}

}

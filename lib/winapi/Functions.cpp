
#include "Functions.hpp"
#include <stdexcept>
#include <vector>
#include <system_error>

namespace winapi {

void throw_error(const std::string& ascii_desc, DWORD error_code) {
	std::error_code ec(error_code, std::system_category());
	throw std::system_error(ec, ascii_desc);
}

std::wstring multi_byte_to_wide_char(const std::string& multi_byte, UINT codepage) {
	if (multi_byte.empty()) {
		return {};
	}
	int num_chars = MultiByteToWideChar(
		codepage,
		MB_ERR_INVALID_CHARS,
		multi_byte.c_str(),
		(int)multi_byte.length(),
		NULL,
		0
	);
	if (num_chars == 0) {
		throw std::runtime_error("MultiByteToWideChar failed (1)");
	}
	std::wstring wide_char;
	wide_char.resize(num_chars);
	int num_chars2 = MultiByteToWideChar(
		codepage,
		MB_ERR_INVALID_CHARS,
		multi_byte.c_str(),
		(int)multi_byte.length(),
		&wide_char[0],
		num_chars
	);
	if (num_chars2 == 0) {
		throw std::runtime_error("MultiByteToWideChar failed (2)");
	}
	return wide_char;
}

std::string wide_char_to_multi_byte(const std::wstring& wide_char, UINT codepage) {
	if (wide_char.empty()) {
		return {};
	}
	int num_bytes = WideCharToMultiByte(
		codepage,
		WC_ERR_INVALID_CHARS,
		wide_char.c_str(),
		(int)wide_char.length(),
		NULL,
		0,
		NULL,
		NULL);
	if (num_bytes == 0) {
		throw std::runtime_error("WideCharToMultiByte failed (1)");
	}
	std::string multi_byte;
	multi_byte.resize(num_bytes);
	int num_bytes2 = WideCharToMultiByte(
		codepage,
		WC_ERR_INVALID_CHARS,
		wide_char.c_str(),
		(int)wide_char.length(),
		&multi_byte[0],
		num_bytes,
		NULL,
		NULL);
	if (num_bytes2 == 0) {
		throw std::runtime_error("WideCharToMultiByte failed (2)");
	}
	return multi_byte;
}

std::wstring utf8_to_utf16(const std::string& multi_byte) {
	return multi_byte_to_wide_char(multi_byte, CP_UTF8);
}

std::wstring acp_to_utf16(const std::string& multi_byte) {
	return multi_byte_to_wide_char(multi_byte, CP_ACP);
}

std::string utf16_to_utf8(const std::wstring& wide_char) {
	return wide_char_to_multi_byte(wide_char, CP_UTF8);
}

std::string get_module_file_name() {
	wchar_t szFileName[MAX_PATH] = {};
	GetModuleFileNameW(NULL, szFileName, MAX_PATH);
	return utf16_to_utf8(szFileName);
}

std::map<std::string, std::string> get_environment_strings() {
	std::map<std::string, std::string> result;
	wchar_t* env = GetEnvironmentStringsW();
	int i = 0;
	while (true) {
		if (env[i] == L'\0') {
			break;
		}
		int j = i;
		while (true) {
			if (env[++j] == L'=') {
				break;
			}
		}
		std::wstring name(env+i, env+j);
		int k = j;
		while (true) {
			if (env[++k] == L'\0') {
				break;
			}
		}
		std::wstring value(env+j+1, env+k);
		result[utf16_to_utf8(name)] = utf16_to_utf8(value);
		i = k+1;
	}
	FreeEnvironmentStrings(env);
	return result;
}

std::wstring expand_environment_strings(const std::wstring& src) {
	std::vector<wchar_t> dst;
	dst.resize(MAX_PATH);
	DWORD result = ExpandEnvironmentStringsW(src.c_str(), dst.data(), dst.size());
	if (result == 0) {
		throw std::runtime_error("ExpandEnvironmentStringsW failed (1)");
	} else if (result > dst.size()) {
		dst.resize(result);
		result = ExpandEnvironmentStringsW(src.c_str(), dst.data(), dst.size());
		if (result == 0) {
			throw std::runtime_error("ExpandEnvironmentStringsW failed (2)");
		}
	}
	return dst.data();
}

std::string expand_environment_strings(const std::string& src) {
	return utf16_to_utf8(expand_environment_strings(utf8_to_utf16(src)));
}

}
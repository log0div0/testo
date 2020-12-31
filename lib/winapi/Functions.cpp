
#include "Functions.hpp"
#include "UTF.hpp"
#include <stdexcept>
#include <vector>

namespace winapi {

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
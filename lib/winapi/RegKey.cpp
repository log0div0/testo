
#include "RegKey.hpp"
#include <stdexcept>
#include "UTF.hpp"

namespace winapi {

RegKey::RegKey(HKEY key, const std::string& path) {
	LSTATUS status = RegOpenKeyEx(
		key,
		utf8_to_utf16(path).c_str(),
		0,
		KEY_ALL_ACCESS,
		&handle
	);
	if (status != ERROR_SUCCESS) {
		throw std::runtime_error("RegOpenKeyEx failed");
	}
}

RegKey::~RegKey() {
	if (handle) {
		RegCloseKey(handle);
		handle = NULL;
	}
}

std::string RegKey::query_str(const std::string& name) const {
	DWORD size = 0;
	DWORD type = REG_NONE;
	LSTATUS status = RegQueryValueEx(handle,
		utf8_to_utf16(name).c_str(),
		NULL,
		&type,
		NULL,
		&size);
	if (status != ERROR_SUCCESS) {
		throw std::runtime_error("RegQueryValueEx failed (1)");
	}
	if (!((type == REG_EXPAND_SZ) || (type == REG_SZ))) {
		throw std::runtime_error("RegQueryValueEx: it's not a string");
	}
	std::wstring value;
	value.resize((size / sizeof(wchar_t)) - 1);
	status = RegQueryValueEx(handle,
		utf8_to_utf16(name).c_str(),
		NULL,
		NULL,
		(uint8_t*)&value[0],
		&size);
	if (status != ERROR_SUCCESS) {
		throw std::runtime_error("RegQueryValueEx failed (2)");
	}
	return utf16_to_utf8(value);
}

void RegKey::set_expand_str(const std::string& name, const std::string& value) {
	std::wstring wvalue = utf8_to_utf16(value);
	LSTATUS status = RegSetValueEx(handle,
		utf8_to_utf16(name).c_str(),
		NULL,
		REG_EXPAND_SZ,
		(uint8_t*)wvalue.c_str(),
		(DWORD)((wvalue.size() + 1) * sizeof(wchar_t))
	);
	if (status != ERROR_SUCCESS) {
		throw std::runtime_error("RegSetValueEx failed");
	}
}
}
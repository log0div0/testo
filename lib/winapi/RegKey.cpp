
#include "RegKey.hpp"
#include "Functions.hpp"
#include <stdexcept>
#include <system_error>

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

std::vector<std::string> RegKey::enum_values() const {
	std::vector<std::string> result;
	std::vector<wchar_t> name;
	while (true) {
		DWORD name_size = 512;
		DWORD type = 0;
		name.resize(name_size);
		LSTATUS status = RegEnumValueW(handle, result.size(),
			name.data(), &name_size,
			NULL,
			&type,
			NULL, NULL);
		if (status == ERROR_MORE_DATA) {
			name.resize(name_size);
			status = RegEnumValueW(handle, result.size(),
				name.data(), &name_size,
				NULL,
				&type,
				NULL, NULL);
			if (status != ERROR_SUCCESS) {
				std::error_code ec(status, std::system_category());
				throw std::system_error(ec, "RegEnumValueW failed (2)");
			}
		} else if (status == ERROR_NO_MORE_ITEMS) {
			break;
		} else if (status != ERROR_SUCCESS) {
			std::error_code ec(status, std::system_category());
			throw std::system_error(ec, "RegEnumValueW failed (1)");
		}
		result.push_back(utf16_to_utf8(name.data()));
	}
	return result;
}

std::string RegKey::get_str(const std::string& name) const {
	DWORD type = 0;
	DWORD value_size = 512;
	std::vector<uint8_t> value;
	value.resize(value_size);
	LSTATUS status = RegGetValueW(handle, NULL,
		utf8_to_utf16(name).c_str(),
		RRF_RT_REG_EXPAND_SZ | RRF_RT_REG_SZ | RRF_NOEXPAND,
		&type,
		value.data(),
		&value_size);
	if (status == ERROR_MORE_DATA) {
		value.resize(value_size);
		status = RegGetValueW(handle, NULL,
			utf8_to_utf16(name).c_str(),
			RRF_RT_REG_EXPAND_SZ | RRF_RT_REG_SZ | RRF_NOEXPAND,
			&type,
			value.data(),
			&value_size);
		if (status != ERROR_SUCCESS) {
			std::error_code ec(status, std::system_category());
			throw std::system_error(ec, "RegGetValueW failed (2)");
		}
	} else if (status != ERROR_SUCCESS) {
		std::error_code ec(status, std::system_category());
		throw std::system_error(ec, "RegGetValueW failed (1)");
	}
	if (!((type == REG_EXPAND_SZ) || (type == REG_SZ))) {
		throw std::runtime_error("RegGetValueW: it's not a string");
	}
	std::wstring str = (wchar_t*)value.data();
	return utf16_to_utf8(expand_environment_strings(str));
}

}


#pragma once

#include <Windows.h>
#include <experimental/filesystem>

namespace winapi {

inline std::experimental::filesystem::path get_module_file_name() {
	TCHAR szFileName[MAX_PATH] = {};
	GetModuleFileName(NULL, szFileName, MAX_PATH);
	return szFileName;
}

inline std::wstring utf8_to_utf16(const std::string& utf8) {
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

inline std::string utf16_to_utf8(const std::wstring& utf16) {
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

struct File {
	File() = default;
	File(const std::string& path, DWORD dwDesiredAccess, DWORD dwCreationDisposition) {
		handle = CreateFile(winapi::utf8_to_utf16(path).c_str(),
			dwDesiredAccess,
			0,
			NULL,
			dwCreationDisposition,
			FILE_ATTRIBUTE_NORMAL,
			NULL);

		if (handle == INVALID_HANDLE_VALUE) {
			throw std::runtime_error("CreateFile failed");
		}
	}

	~File() {
		if (handle) {
			CloseHandle(handle);
		}
	}

	File(File&& other): handle(other.handle) {
		other.handle = NULL;
	}

	File& operator=(File&& other) {
		std::swap(handle, other.handle);
		return *this;
	}

	size_t read(uint8_t* data, size_t size) const {
		DWORD result = 0;
		bool success = ReadFile(handle, data, (DWORD)size, &result, NULL);
		if (!success) {
			throw std::runtime_error("ReadFile failed");
		}
		return result;
	}

	size_t write(const uint8_t* data, size_t size) {
		DWORD result = 0;
		bool success = WriteFile(handle, data, (DWORD)size, &result, NULL);
		if (!success) {
			throw std::runtime_error("WriteFile failed");
		}
		return result;
	}

	size_t size() const {
		DWORD high = 0;
		DWORD low = GetFileSize(handle, &high);
		if (low == INVALID_FILE_SIZE) {
			throw std::runtime_error("GetFileSize failed");
		}
		return (size_t(high) << 32) | low;
	}

	HANDLE handle = NULL;
};

struct RegKey {
	RegKey(HKEY key, const std::string& path) {
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
	~RegKey() {
		if (handle) {
			RegCloseKey(handle);
			handle = NULL;
		}
	}

	std::string query_str(const std::string& name) const {
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

	void set_expand_str(const std::string& name, const std::string& value) {
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

	RegKey(RegKey&&);
	RegKey& operator=(RegKey&&);

private:
	HKEY handle = NULL;
};

struct Service {
	Service(SC_HANDLE handle_): handle(handle_) {

	}

	~Service() {
		if (handle) {
			CloseServiceHandle(handle);
		}
	}

	Service(Service&& other);
	Service& operator=(Service&& other);

	void start() {
		if (!StartService(handle, 0, NULL)) {
			throw std::runtime_error("StartService failed");
		}
	}

	SERVICE_STATUS queryStatus() {
		SERVICE_STATUS status = {};
		if (!QueryServiceStatus(handle, &status)) {
			throw std::runtime_error("QueryServiceStatus failed");
		}
		return status;
	}

private:
	SC_HANDLE handle = NULL;
};

struct SCManager {
	SCManager() {
		handle = OpenSCManager(NULL, NULL, SC_MANAGER_CREATE_SERVICE);
		if (!handle) {
			throw std::runtime_error("OpenSCManager failed");
		}
	}

	~SCManager() {
		if (handle) {
			CloseServiceHandle(handle);
		}
	}

	SCManager(SCManager&& other);
	SCManager& operator=(SCManager&& other);

	Service service(const std::string& name) {
		SC_HANDLE hService = OpenService(handle, utf8_to_utf16(name).c_str(), SERVICE_QUERY_STATUS | SERVICE_START);
		if (!hService) {
			throw std::runtime_error("OpenServiceA failed");
		}
		return hService;
	}

private:
	SC_HANDLE handle = NULL;
};

}

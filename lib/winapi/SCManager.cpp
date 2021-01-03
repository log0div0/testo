
#include "SCManager.hpp"
#include <stdexcept>
#include "Functions.hpp"

namespace winapi {

SCManager::SCManager() {
	handle = OpenSCManager(NULL, NULL, SC_MANAGER_CREATE_SERVICE);
	if (!handle) {
		throw std::runtime_error("OpenSCManager failed");
	}
}

SCManager::~SCManager() {
	if (handle) {
		CloseServiceHandle(handle);
	}
}

Service SCManager::service(const std::string& name, DWORD dwDesiredAccess) {
	SC_HANDLE hService = OpenService(handle, utf8_to_utf16(name).c_str(), dwDesiredAccess);
	if (!hService) {
		throw std::runtime_error("OpenServiceA failed");
	}
	return hService;
}

}

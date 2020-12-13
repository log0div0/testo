
#include "Service.hpp"
#include <stdexcept>

namespace winapi {

Service::Service(SC_HANDLE handle_): handle(handle_) {

}

Service::~Service() {
	if (handle) {
		CloseServiceHandle(handle);
	}
}

void Service::start() {
	if (!StartService(handle, 0, NULL)) {
		throw std::runtime_error("StartService failed");
	}
}

SERVICE_STATUS Service::control(DWORD signal) {
	SERVICE_STATUS status = {};
	if (!ControlService(handle, signal, &status)) {
		throw std::runtime_error("ControlService failed");
	}
	return status;
}

SERVICE_STATUS Service::queryStatus() const {
	SERVICE_STATUS status = {};
	if (!QueryServiceStatus(handle, &status)) {
		throw std::runtime_error("QueryServiceStatus failed");
	}
	return status;
}

}

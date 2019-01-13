
#include "guest_process.hpp"
#include <stdexcept>
#include <sstream>
#include "throw_if_failed.hpp"
#include "string.hpp"
#include "safe_array.hpp"

namespace vbox {

GuestProcess::GuestProcess(IGuestProcess* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

GuestProcess::~GuestProcess() {
	if (handle) {
		IGuestProcess_Release(handle);
	}
}

GuestProcess::GuestProcess(GuestProcess&& other): handle(other.handle) {
	other.handle = nullptr;
}

GuestProcess& GuestProcess::operator=(GuestProcess&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::vector<uint8_t> GuestProcess::read(uint32_t handle, uint32_t bytes_to_read, uint32_t timeout_ms) {
	try {
		SafeArray safe_array;
		throw_if_failed(IGuestProcess_Read(this->handle, handle, bytes_to_read, timeout_ms, SAFEARRAY_AS_OUT_PARAM(uint8_t, safe_array)));
		ArrayOut array_out = safe_array.copy_out(VT_UI1);
		std::vector<uint8_t> result(array_out.begin(), array_out.end());
		result.push_back(0);
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

ProcessWaitResult GuestProcess::wait_for(ProcessWaitForFlag wait_flag, uint32_t timeout_ms) {
	try {
		ProcessWaitResult_T result = ProcessWaitResult_None;
		throw_if_failed(IGuestProcess_WaitFor(handle, wait_flag, timeout_ms, &result));
		return (ProcessWaitResult)result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

ProcessWaitResult GuestProcess::wait_for(const std::vector<ProcessWaitForFlag>& flags, uint32_t timeout_ms) {
	try {
		int bitset = 0;
		for (auto flag: flags) {
			bitset |= flag;
		}

		SafeArray safe_array = SafeArray::bitset(bitset);
		ProcessWaitResult_T result = ProcessWaitResult_None;
		throw_if_failed(IGuestProcess_WaitForArray(handle,
			SAFEARRAY_AS_IN_PARAM(ProcessWaitForFlag_T, safe_array),
			timeout_ms, &result));

		return (ProcessWaitResult)result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

LONG GuestProcess::exit_code() const {
	try {
		LONG result = 0;
		throw_if_failed(IGuestProcess_get_ExitCode(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

ProcessStatus GuestProcess::status() const {
	try {
		ProcessStatus_T result = ProcessStatus_Undefined;
		IGuestProcess_get_Status(handle, &result);
		return (ProcessStatus) result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool GuestProcess::is_alive() const {
	auto state = status();
	return ((state == ProcessStatus_Starting) ||
		(state == ProcessStatus_Started) ||
		(state == ProcessStatus_Paused) ||
		(state == ProcessStatus_Terminating));
}

}

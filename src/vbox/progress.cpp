
#include "progress.hpp"
#include <stdexcept>
#include <sstream>
#include "throw_if_failed.hpp"
#include "string.hpp"

namespace vbox {

Progress::Progress(IProgress* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Progress::~Progress() {
	if (handle) {
		IProgress_Release(handle);
	}
}

Progress::Progress(Progress&& other): handle(other.handle) {
	other.handle = nullptr;
}

Progress& Progress::operator=(Progress&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Progress::description() const {
	try {
		BSTR result = nullptr;
		throw_if_failed(IProgress_get_Description(handle, &result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Progress::wait_for_completion(long timeout) const {
	try {
		throw_if_failed(IProgress_WaitForCompletion(handle, timeout));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Progress::wait_and_throw_if_failed() const {
	try {
		wait_for_completion();

		LONG rc = result_code();
		if (rc) {
			vbox::VirtualBoxErrorInfo error = error_info();
			if (error) {
				auto error_text = error.text();
				throw std::runtime_error(error_text);
			} else {
				std::stringstream ss;
				ss << "Error code: " << std::hex << rc << std::dec;
				throw std::runtime_error(ss.str());
			}
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(description()));
	}
}

LONG Progress::result_code() const {
	try {
		LONG result = 0;
		throw_if_failed(IProgress_get_ResultCode(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

VirtualBoxErrorInfo Progress::error_info() const {
	try {
		IVirtualBoxErrorInfo* result = nullptr;
		throw_if_failed(IProgress_get_ErrorInfo(handle, &result));
		if (result) {
			return result;
		} else {
			return {};
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}

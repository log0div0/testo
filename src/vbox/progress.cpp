
#include "progress.hpp"
#include <stdexcept>
#include "error.hpp"
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
		HRESULT rc = IProgress_get_Description(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Progress::wait_for_completion(long timeout) const {
	try {
		HRESULT rc = IProgress_WaitForCompletion(handle, timeout);
		if (FAILED(rc)) {
			throw Error(rc);
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}

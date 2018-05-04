
#include "machine.hpp"
#include <stdexcept>
#include "error.hpp"

namespace vbox {

Machine::Machine(IMachine* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Machine::~Machine() {
	if (handle) {
		IMachine_Release(handle);
	}
}

Machine::Machine(Machine&& other): handle(other.handle) {
	other.handle = nullptr;
}

Machine& Machine::operator=(Machine&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Machine::name() const {
	try {
		BSTR name = nullptr;
		HRESULT rc = IMachine_get_Name(handle, &name);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}

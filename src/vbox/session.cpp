
#include "session.hpp"
#include <stdexcept>
#include "throw_if_failed.hpp"
#include "string.hpp"

namespace vbox {

Session::Session(ISession* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Session::~Session() {
	if (handle) {
		ISession_Release(handle);
	}
}

Session::Session(Session&& other): handle(other.handle) {
	other.handle = nullptr;
}

Session& Session::operator=(Session&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Session::name() const {
	try {
		BSTR name = nullptr;
		throw_if_failed(ISession_get_Name(handle, &name));
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Machine Session::machine() const {
	try {
		IMachine* machine = nullptr;
		throw_if_failed(ISession_get_Machine(handle, &machine));
		return machine;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Session::unlock_machine() {
	try {
		throw_if_failed(ISession_UnlockMachine(handle));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}

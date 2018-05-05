
#include "session.hpp"
#include <stdexcept>
#include "error.hpp"
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
		HRESULT rc = ISession_get_Name(handle, &name);
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
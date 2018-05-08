
#include "display.hpp"
#include "throw_if_failed.hpp"

namespace vbox {

Display::Display(IDisplay* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Display::~Display() {
	if (handle) {
		IDisplay_Release(handle);
	}
}

Display::Display(Display&& other): handle(other.handle) {
	other.handle = nullptr;
}

Display& Display::operator=(Display&& other) {
	std::swap(handle, other.handle);
	return *this;
}

}

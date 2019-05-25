
#include "keyboard.hpp"
#include "safe_array.hpp"
#include "throw_if_failed.hpp"

#include <algorithm>

namespace vbox {

Keyboard::Keyboard(IKeyboard* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Keyboard::~Keyboard() {
	if (handle) {
		IKeyboard_Release(handle);
	}
}

Keyboard::Keyboard(Keyboard&& other): handle(other.handle) {
	other.handle = nullptr;
}

Keyboard& Keyboard::operator=(Keyboard&& other) {
	std::swap(handle, other.handle);
	return *this;
}

void Keyboard::putScancode(uint32_t code) {
	try {
		throw_if_failed(IKeyboard_PutScancode(handle, code));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}

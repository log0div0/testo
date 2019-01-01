
#include "console.hpp"
#include "throw_if_failed.hpp"

namespace vbox {

Console::Console(IConsole* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Console::~Console() {
	if (handle) {
		IConsole_Release(handle);
	}
}

Console::Console(Console&& other): handle(other.handle) {
	other.handle = nullptr;
}

Console& Console::operator=(Console&& other) {
	std::swap(handle, other.handle);
	return *this;
}

Progress Console::power_up() const {
	try {
		IProgress* result = nullptr;
		throw_if_failed(IConsole_PowerUp(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress Console::power_down() const {
	try {
		IProgress* result = nullptr;
		throw_if_failed(IConsole_PowerDown(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Console::pause() const {
	try {
		throw_if_failed(IConsole_Pause(handle));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Console::resume() const {
	try {
		throw_if_failed(IConsole_Resume(handle));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}


Display Console::display() const {
	try {
		IDisplay* result = nullptr;
		throw_if_failed(IConsole_get_Display(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Keyboard Console::keyboard() const {
	try {
		IKeyboard* result = nullptr;
		throw_if_failed(IConsole_get_Keyboard(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Guest Console::guest() const {
	try {
		IGuest* result = nullptr;
		throw_if_failed(IConsole_get_Guest(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}

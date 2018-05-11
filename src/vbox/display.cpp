
#include "display.hpp"
#include "throw_if_failed.hpp"
#include "string.hpp"

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

std::string Display::attach_framebuffer(ULONG screen_id, const Framebuffer& framebuffer) {
	BSTR result = nullptr;
	throw_if_failed(IDisplay_AttachFramebuffer(handle, screen_id, framebuffer.handle, &result));
	return StringOut(result);
}

void Display::detach_framebuffer(ULONG screen_id, const std::string& uuid) {
	throw_if_failed(IDisplay_DetachFramebuffer(handle, screen_id, StringIn(uuid)));
}

}

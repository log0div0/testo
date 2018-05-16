
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

void Display::get_screen_resolution(ULONG screen_id,
	ULONG* width,
	ULONG* height,
	ULONG* bits_per_pixel,
	LONG* x_origin,
	LONG* y_origin,
	GuestMonitorStatus* guest_monitor_status
) const {
	try {
		throw_if_failed(IDisplay_GetScreenResolution(handle, screen_id,
			width,
			height,
			bits_per_pixel,
			x_origin,
			y_origin,
			(GuestMonitorStatus_T*)guest_monitor_status
		));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

SafeArray Display::take_screen_shot_to_array(ULONG screen_id, ULONG width, ULONG height, BitmapFormat bitmap_format) const {
	try {
		SafeArray result;
		throw_if_failed(IDisplay_TakeScreenShotToArray(handle, screen_id,
			width,
			height,
			bitmap_format,
			SAFEARRAY_AS_OUT_PARAM(uint8_t, result)
		));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}

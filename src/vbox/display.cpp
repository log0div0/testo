
#include "display.hpp"
#include "throw_if_failed.hpp"

namespace vbox {

std::ostream& operator<<(std::ostream& stream, const ScreenResolution& screen_resolution) {
	stream
		<< "width=" << screen_resolution.width
		<< " height=" << screen_resolution.height
		<< " bits_per_pixel=" << screen_resolution.bits_per_pixel
		<< " x_origin=" << screen_resolution.x_origin
		<< " y_origin=" << screen_resolution.y_origin
		<< " guest_monitor_status=" << screen_resolution.guest_monitor_status
	;
	return stream;
}

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

ScreenResolution Display::get_screen_resolution(ULONG screen_id) const {
	try {
		ScreenResolution result;
		throw_if_failed(IDisplay_GetScreenResolution(handle, screen_id,
			&result.width,
			&result.height,
			&result.bits_per_pixel,
			&result.x_origin,
			&result.y_origin,
			(GuestMonitorStatus_T*)&result.guest_monitor_status
		));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}

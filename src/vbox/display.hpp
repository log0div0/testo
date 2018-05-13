
#pragma once

#include "api.hpp"

namespace vbox {

struct ScreenResolution {
	ULONG width = 0;
	ULONG height = 0;
	ULONG bits_per_pixel = 0;
	LONG x_origin = 0;
	LONG y_origin = 0;
	GuestMonitorStatus guest_monitor_status = GuestMonitorStatus_Disabled;
};

std::ostream& operator<<(std::ostream& stream, const ScreenResolution& screen_resolution);

struct Display {
	Display(IDisplay* handle);
	~Display();

	Display(const Display&) = delete;
	Display& operator=(const Display&) = delete;
	Display(Display&& other);
	Display& operator=(Display&& other);

	ScreenResolution get_screen_resolution(ULONG screen_id = 0) const;

	IDisplay* handle = nullptr;
};

}

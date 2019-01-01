
#pragma once

#include "safe_array.hpp"

namespace vbox {

struct Display {
	Display() = default;
	Display(IDisplay* handle);
	~Display();

	Display(const Display&) = delete;
	Display& operator=(const Display&) = delete;
	Display(Display&& other);
	Display& operator=(Display&& other);

	void get_screen_resolution(ULONG screen_id,
		ULONG* width,
		ULONG* height,
		ULONG* bits_per_pixel,
		LONG* x_origin,
		LONG* y_origin,
		GuestMonitorStatus* guest_monitor_status
	) const;
	SafeArray take_screen_shot_to_array(ULONG screen_id, ULONG width, ULONG height, BitmapFormat bitmap_format) const;

	IDisplay* handle = nullptr;
};

}

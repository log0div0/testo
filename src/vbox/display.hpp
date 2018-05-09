
#pragma once

#include "framebuffer.hpp"

namespace vbox {

struct Display {
	Display(IDisplay* handle);
	~Display();

	Display(const Display&) = delete;
	Display& operator=(const Display&) = delete;
	Display(Display&& other);
	Display& operator=(Display&& other);

	std::string attach_framebuffer(ULONG screen_id, const Framebuffer& framebuffer);

	IDisplay* handle = nullptr;
};

}

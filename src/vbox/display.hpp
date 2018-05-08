
#pragma once

#include "api.hpp"

namespace vbox {

struct Display {
	Display(IDisplay* handle);
	~Display();

	Display(const Display&) = delete;
	Display& operator=(const Display&) = delete;
	Display(Display&& other);
	Display& operator=(Display&& other);

	IDisplay* handle = nullptr;
};

}

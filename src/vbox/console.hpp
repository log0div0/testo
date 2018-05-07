
#pragma once

#include "progress.hpp"

namespace vbox {

struct Console {
	Console(IConsole* handle);
	~Console();

	Console(const Console&) = delete;
	Console& operator=(const Console&) = delete;
	Console(Console&& other);
	Console& operator=(Console&& other);

	Progress power_up() const;

	IConsole* handle = nullptr;
};

}

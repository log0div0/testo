
#pragma once

#include "progress.hpp"
#include "display.hpp"

namespace vbox {

struct Console {
	Console() = default;
	Console(IConsole* handle);
	~Console();

	Console(const Console&) = delete;
	Console& operator=(const Console&) = delete;
	Console(Console&& other);
	Console& operator=(Console&& other);

	Progress power_up() const;
	Progress power_down() const;

	Display display() const;

	IConsole* handle = nullptr;
};

}

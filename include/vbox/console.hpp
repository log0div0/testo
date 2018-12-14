
#pragma once

#include "progress.hpp"
#include "display.hpp"
#include "keyboard.hpp"
#include "guest.hpp"

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

	void pause() const;
	void resume() const;

	Display display() const;
	Keyboard keyboard() const;
	Guest guest() const;

	IConsole* handle = nullptr;
};

}

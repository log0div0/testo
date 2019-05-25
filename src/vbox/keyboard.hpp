
#pragma once

#include "enums.hpp"
#include <unordered_map>
#include <vector>

namespace vbox {

struct Keyboard {
	Keyboard(IKeyboard* handle);
	~Keyboard();

	Keyboard(const Keyboard&) = delete;
	Keyboard& operator=(const Keyboard&) = delete;
	Keyboard(Keyboard&& other);
	Keyboard& operator=(Keyboard&& other);

	void putScancode(uint32_t code);

	IKeyboard* handle = nullptr;
};

}
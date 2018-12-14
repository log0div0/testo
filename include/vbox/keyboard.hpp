
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

	void putScancodes(const std::vector<std::string>& buttons);
	void releaseKeys(const std::vector<std::string>& buttons);

	IKeyboard* handle = nullptr;
	std::unordered_map<std::string, std::vector<uint32_t>> scancodes;
};

}
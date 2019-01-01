
#include "keyboard.hpp"
#include "safe_array.hpp"
#include "throw_if_failed.hpp"

#include <algorithm>

namespace vbox {

Keyboard::Keyboard(IKeyboard* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}

	scancodes.insert({
		{"ESC", {1}},
		{"ONE", {2}},
		{"TWO", {3}},
		{"THREE", {4}},
		{"FOUR", {5}},
		{"FIVE", {6}},
		{"SIX", {7}},
		{"SEVEN", {8}},
		{"EIGHT", {9}},
		{"NINE", {10}},
		{"ZERO", {11}},
		{"MINUS", {12}},
		{"EQUAL", {13}},
		{"BACKSPACE", {14}},
		{"TAB", {15}},
		{"Q", {16}},
		{"W", {17}},
		{"E", {18}},
		{"R", {19}},
		{"T", {20}},
		{"Y", {21}},
		{"U", {22}},
		{"I", {23}},
		{"O", {24}},
		{"P", {25}},
		{"LEFTBRACE", {26}},
		{"RIGHTBRACE", {27}},
		{"ENTER", {28}},
		{"LEFTCTRL", {29}},
		{"A", {30}},
		{"S", {31}},
		{"D", {32}},
		{"F", {33}},
		{"G", {34}},
		{"H", {35}},
		{"J", {36}},
		{"K", {37}},
		{"L", {38}},
		{"SEMICOLON", {39}},
		{"APOSTROPHE", {40}},
		{"GRAVE", {41}},
		{"LEFTSHIFT", {42}},
		{"BACKSLASH", {43}},
		{"Z", {44}},
		{"X", {45}},
		{"C", {46}},
		{"V", {47}},
		{"B", {48}},
		{"N", {49}},
		{"M", {50}},
		{"COMMA", {51}},
		{"DOT", {52}},
		{"SLASH", {53}},
		{"RIGHTSHIFT", {54}},
		{"LEFTALT", {56}},
		{"SPACE", {57}},
		{"CAPSLOCK", {58}},
		{"NUMLOCK", {69}}, //TODO: recheck
		{"SCROLLLOCK", {70}},

		{"RIGHTCTRL", {97}},
		{"RIGHTALT", {100}},

		{"HOME", {224,71}},
		{"UP", {224, 72}},
		{"PAGEUP", {224,73}},
		{"LEFT", {224,75}},
		{"RIGHT", {224,77}},
		{"END", {224,79}},
		{"DOWN", {224,80}},
		{"PAGEDOWN", {224,81}},
		{"INSERT", {224,82}},
		{"DELETE", {224,83}},

		{"SCROLLUP", {177}},
		{"SCROLLDOWN", {178}},
	});
}

Keyboard::~Keyboard() {
	if (handle) {
		IKeyboard_Release(handle);
	}
}

Keyboard::Keyboard(Keyboard&& other): handle(other.handle) {
	other.handle = nullptr;
}

Keyboard& Keyboard::operator=(Keyboard&& other) {
	std::swap(handle, other.handle);
	return *this;
}

void Keyboard::putScancodes(const std::vector<std::string>& buttons) {
	try {
		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);
			auto codes = scancodes.find(button);
			if (codes == scancodes.end()) {
				throw std::runtime_error("Unknown scancode");
			}

			for (auto code: codes->second) {
				throw_if_failed(IKeyboard_PutScancode(handle, code));
			}
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Keyboard::releaseKeys(const std::vector<std::string>& buttons) {
	try {
		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);
			auto codes = scancodes.find(button);
			if (codes == scancodes.end()) {
				throw std::runtime_error("Unknown scancode");
			}

			for (auto code: codes->second) {
				throw_if_failed(IKeyboard_PutScancode(handle, code | 0x80));
			}
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}


}

#pragma once

#include <string>

enum MouseButton {
	None = 0,
	Left = 1,
	Right = 2,
	Middle = 3,
	WheelUp = 4,
	WheelDown = 5
};

std::string ToString(MouseButton button);
MouseButton ToMouseButton(const std::string& button);
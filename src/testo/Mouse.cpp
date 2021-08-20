
#include "Mouse.hpp"
#include <map>
#include <vector>

std::vector<std::string> mb_to_str = {
	"invalid",
	"lbtn",
	"rbtn",
	"mbtn",
	"up",
	"down"
};

std::map<std::string, MouseButton> str_to_mb = {
	{"invalid", None},
	{"lbtn", Left},
	{"rbtn", Right},
	{"mbtn", Middle},
	{"up", WheelUp},
	{"down", WheelDown},
};

std::string ToString(MouseButton button) {
	return mb_to_str.at((int)button);
}

MouseButton ToMouseButton(const std::string& button) {
	return str_to_mb.at(button);
}


#pragma once

#include "Rect.hpp"
#include <string>
#include <vector>

namespace nn {

struct Homm3Object {
	static const std::vector<std::string> classes_names;
	static bool check_class_name(const std::string& name);

	std::string class_name;
	Rect rect;
};

}

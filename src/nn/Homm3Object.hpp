
#pragma once

#include "Rect.hpp"

namespace nn {

struct Homm3Object {
	static bool check_class_id(const std::string& id);

	std::string class_id;
	Rect rect;
};

}

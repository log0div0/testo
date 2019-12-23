
#pragma once

#include "Rect.hpp"
#include <string>

namespace nn {

struct Word {
	Rect rect;
	std::string text;
};

}

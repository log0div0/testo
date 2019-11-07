
#pragma once

#include <darknet/Image.hpp>
#include <string>

namespace screen_selection {

std::string text_to_query(const std::string& text);

struct ScreenSelector {
	bool exec(stb::Image& image, const std::string& query);
};

}

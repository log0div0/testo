
#pragma once

#include "../Image.hpp"

struct TextDetector {
	TextDetector();
	bool detect(Image& image, const std::string& text,
		const std::string& foreground,
		const std::string& background);
};

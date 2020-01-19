
#pragma once

#include "TextDetector.hpp"
#include "TextRecognizer.hpp"
#include "TextLine.hpp"
#include <map>

namespace nn {

struct OCRResult {
	std::vector<Rect> search(const std::string& query, const std::string& fg_color = {}, const std::string& bg_color = {});
	std::vector<TextLine> textlines;
};

struct OCR {
	static OCR& instance();

	OCRResult run(const stb::Image& image);

private:
	OCR() = default;

	TextDetector detector;
	TextRecognizer recognizer;
};

}

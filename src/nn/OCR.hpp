
#pragma once

#include "TextDetector.hpp"
#include "TextRecognizer.hpp"
#include "TextLine.hpp"
#include <map>

namespace nn {

struct OCRResult {
	std::vector<Rect> search(const std::string& query);
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

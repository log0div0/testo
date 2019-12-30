
#pragma once

#include "TextDetector.hpp"
#include "TextRecognizer.hpp"
#include "TextLine.hpp"

namespace nn {

struct OCR {
	std::vector<TextLine> run(const stb::Image& image);

private:
	TextDetector detector;
	TextRecognizer recognizer;
};

}

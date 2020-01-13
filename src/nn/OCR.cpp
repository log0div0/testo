
#include "OCR.hpp"
#include <iostream>
#include <algorithm>

namespace nn {

std::vector<TextLine> OCR::run(const stb::Image& image) {
	std::vector<TextLine> textlines = detector.detect(image);
	// recognizer.recognize(image, textlines);
	return textlines;
}

}

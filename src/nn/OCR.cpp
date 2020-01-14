
#include "OCR.hpp"
#include <iostream>
#include <algorithm>

namespace nn {

std::vector<TextLine> OCR::run(const stb::Image& image) {
	std::vector<TextLine> textlines = detector.detect(image);
	for (auto& textline: textlines) {
		for (auto& word: textline.words) {
			recognizer.recognize(image, word);
		}
	}
	return textlines;
}

}


#include "OCR.hpp"
#include <iostream>
#include <algorithm>

namespace nn {

OCR& OCR::instance() {
	static OCR ocr;
	return ocr;
}

OCRResult OCR::run(const stb::Image& image) {
	OCRResult result;
	result.textlines = detector.detect(image);
	for (auto& textline: result.textlines) {
		for (auto& word: textline.words) {
			recognizer.recognize(image, word);
		}
	}
	return result;
}

}

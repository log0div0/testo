
#include "OCR.hpp"
#include <iostream>
#include <algorithm>
#include "TextDetector.hpp"
#include "TextRecognizer.hpp"

namespace nn {

Tensor find_text(const stb::Image<stb::RGB>* image) {
	Tensor result;

	result.textlines = TextDetector::instance().detect(image);

	for (auto& textline: result.textlines) {
		TextRecognizer::instance().recognize(image, textline);
	}

	std::sort(result.textlines.begin(), result.textlines.end(), [](const TextLine& a, const TextLine& b) {
		return a.rect.top < b.rect.top;
	});

	return result;
}

Tensor find_img(const stb::Image<stb::RGB>* image, const fs::path& path_to_img) {
	throw std::runtime_error("Todo");
}

}

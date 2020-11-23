
#include "OCR.hpp"
#include <iostream>
#include <algorithm>
#include "TextDetector.hpp"
#include "TextRecognizer.hpp"
#include "ImgDetector.hpp"

namespace nn {

TextTensor find_text(const stb::Image<stb::RGB>* image) {
	TextTensor result;

	result.objects = TextDetector::instance().detect(image);

	for (auto& textline: result.objects) {
		TextRecognizer::instance().recognize(image, textline);
	}

	std::sort(result.objects.begin(), result.objects.end(), [](const TextLine& a, const TextLine& b) {
		return a.rect.top < b.rect.top;
	});

	return result;
}

ImgTensor find_img(const stb::Image<stb::RGB>* image, const fs::path& path_to_img) {
	ImgTensor result;

	result.objects = ImgDetector::instance().detect(image, path_to_img);

	std::sort(result.objects.begin(), result.objects.end(), [](const Img& a, const Img& b) {
		return a.rect.top < b.rect.top;
	});

	return result;
}

}

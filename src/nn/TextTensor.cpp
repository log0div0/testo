
#include "TextTensor.hpp"
#include "TextDetector.hpp"
#include "TextRecognizer.hpp"
#include <algorithm>

namespace nn {

TextTensor TextTensor::match(const std::string& text) {
	TextTensor result;
	for (auto& textline: objects) {
		for (auto& new_textline: textline.match(text)) {
			result.objects.push_back(new_textline);
		}
	}
	return result;
}

TextTensor TextTensor::match_foreground(const stb::Image<stb::RGB>* image, const std::string& color) {
	TextTensor result;
	for (auto& textline: objects) {
		if (textline.match_foreground(image, color)) {
			result.objects.push_back(textline);
		}
	}
	return result;
}

TextTensor TextTensor::match_background(const stb::Image<stb::RGB>* image, const std::string& color) {
	TextTensor result;
	for (auto& textline: objects) {
		if (textline.match_background(image, color)) {
			result.objects.push_back(textline);
		}
	}
	return result;
}

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

}

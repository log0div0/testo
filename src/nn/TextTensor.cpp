
#include "TextTensor.hpp"
#include "TextDetector.hpp"
#include "TextRecognizer.hpp"
#include "TextColorPicker.hpp"
#include <algorithm>

namespace nn {

TextTensor TextTensor::match_text(const stb::Image<stb::RGB>* image, const std::string& text) {
	TextTensor result;
	for (auto& textline: objects) {
		for (auto& new_textline: TextRecognizer::instance().recognize(image, textline, text)) {
			result.objects.push_back(new_textline);
		}
	}
	return result;
}

TextTensor TextTensor::match_color(const stb::Image<stb::RGB>* image, const std::string& fg, const std::string& bg) {
	TextTensor result;
	for (auto& textline: objects) {
		if (TextColorPicker::instance().run(image, textline, fg, bg)) {
			result.objects.push_back(textline);
		}
	}
	return result;
}

TextTensor find_text(const stb::Image<stb::RGB>* image) {
	TextTensor result;

	std::vector<TextLine> words = TextDetector::instance().detect(image);

	std::sort(words.begin(), words.end(), [](const TextLine& a, const TextLine& b) {
		return a.rect.left < b.rect.left;
	});

	std::vector<bool> visited_words(words.size(), false);
	for (size_t i = 0; i < words.size(); ++i) {
		if (visited_words[i]) {
			continue;
		}
		visited_words[i] = true;
		size_t a = i;

		TextLine textline;
		textline.rect = words[a].rect;

		while (true) {
textline_next:
			for (size_t j = a + 1; j < words.size(); ++j) {
				if (visited_words[j]) {
					continue;
				}
				size_t b = j;

				if (words[b].rect.left > (words[a].rect.right + words[a].rect.height())) {
					goto textline_finish;
				}
				int32_t mean_height = (words[a].rect.height() + words[b].rect.height()) / 2;
				int32_t min_bottom = std::min(words[a].rect.bottom, words[b].rect.bottom);
				int32_t max_top = std::max(words[a].rect.top, words[b].rect.top);
				if ((min_bottom - max_top) >= (mean_height / 2)) {
					visited_words[j] = true;
					textline.rect |= words[b].rect;
					a = b;
					goto textline_next;
				}
			}
			goto textline_finish;
		}
textline_finish:
		result.objects.push_back(textline);
	}

	std::sort(result.objects.begin(), result.objects.end(), [](const TextLine& a, const TextLine& b) {
		return a.rect.top < b.rect.top;
	});

	return result;
}

}

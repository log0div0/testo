
#include "OCR.hpp"
#include <utf8.hpp>
#include <iostream>
#include <algorithm>

namespace nn {

std::vector<Rect> OCRResult::search(const std::string& query_str, const std::string& fg_color, const std::string& bg_color) {
	std::vector<std::string> query;
	for (auto& char_: utf8::split_to_chars(query_str)) {
		if (char_ == " ") {
			continue;
		}
		query.push_back(char_);
	}
	std::vector<Rect> result;
	for (auto& textline: textlines) {
		for (auto& rect: textline.search(query)) {
			result.push_back(rect);
		}
	}
	return result;
}

OCR& OCR::instance() {
	static OCR ocr;
	return ocr;
}

OCRResult OCR::run(const stb::Image& image) {
	std::vector<Word> words = detector.detect(image);

	std::sort(words.begin(), words.end(), [](const Word& a, const Word& b) {
		return a.rect.left < b.rect.left;
	});

	std::vector<bool> visited_words(words.size(), false);
	std::vector<TextLine> textlines;
	for (size_t i = 0; i < words.size(); ++i) {
		if (visited_words[i]) {
			continue;
		}
		visited_words[i] = true;
		size_t a = i;

		TextLine textline;
		textline.rect = words[a].rect;
		for (auto& char_: recognizer.recognize(image, words[a])) {
			textline.chars.push_back(char_);
		}
		textline.words.push_back(words[a]);

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
					for (auto& char_: recognizer.recognize(image, words[b])) {
						textline.chars.push_back(char_);
					}
					textline.words.push_back(words[b]);
					a = b;
					goto textline_next;
				}
			}
			goto textline_finish;
		}
textline_finish:
		textlines.push_back(textline);
	}

	std::sort(textlines.begin(), textlines.end(), [](const TextLine& a, const TextLine& b) {
		return a.rect.top < b.rect.top;
	});

	OCRResult result;
	result.textlines = std::move(textlines);
	return result;
}

}

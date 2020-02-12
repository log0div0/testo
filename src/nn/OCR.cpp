
#include "OCR.hpp"
#include <utf8.hpp>
#include <iostream>
#include <algorithm>
#include "TextDetector.hpp"
#include "TextRecognizer.hpp"
#include "TextColorPicker.hpp"

namespace nn {

bool Char::match(const std::string& query) {
	for (auto& code: codes) {
		if (code == query) {
			return true;
		}
	}
	return false;
}

bool Char::matchColor(const std::string& color) {
	if (!color.size()) {
		return true;
	}
	if (!this->color.size()) {
		TextColorPicker::instance().run(*this);
	}
	return this->color == color;
}

bool Char::matchBackgroundColor(const std::string& color) {
	if (!color.size()) {
		return true;
	}
	if (!this->backgroundColor.size()) {
		TextColorPicker::instance().run(*this);
	}
	return this->backgroundColor == color;
}

std::vector<Rect> TextLine::search(const std::vector<std::string>& query, const std::string& color, const std::string& backgroundColor) {
	if (chars.size() < query.size()) {
		return {};
	}
	std::vector<Rect> result;
	for (size_t i = 0; i < (chars.size() - query.size() + 1); ++i) {
		bool match = true;
		for (size_t j = 0; j < query.size(); ++j) {
			if (!chars[i + j].match(query[j])) {
				match = false;
				break;
			}
		}
		if (!match) {
			continue;
		}
		size_t matchColor = 0;
		size_t matchBackgroundColor = 0;
		for (size_t j = 0; j < query.size(); ++j) {
			if (chars[i + j].matchColor(color)) {
				++matchColor;
			}
			if (chars[i + j].matchBackgroundColor(backgroundColor)) {
				++matchBackgroundColor;
			}
		}
		if (matchColor < (query.size() / 2)) {
			continue;
		}
		if (matchBackgroundColor < (query.size() / 2)) {
			continue;
		}
		Rect rect = chars[i].rect;
		for (size_t j = 1; j < query.size(); ++j) {
			rect |= chars[i + j].rect;
		}
		result.push_back(rect);
	}
	return result;
}

std::vector<Rect> OCR::search(const std::string& query_str, const std::string& color, const std::string& backgroundColor) {
	std::vector<std::string> query;
	for (auto& char_: utf8::split_to_chars(query_str)) {
		if (char_ == " ") {
			continue;
		}
		query.push_back(char_);
	}
	std::vector<Rect> result;
	for (auto& textline: textlines) {
		for (auto& rect: textline.search(query, color, backgroundColor)) {
			result.push_back(rect);
		}
	}
	return result;
}

OCR::OCR(const stb::Image* image_): image(image_) {
	std::vector<Word> words = TextDetector::instance().detect(image);

	std::sort(words.begin(), words.end(), [](const Word& a, const Word& b) {
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
		textline.image = image;
		textline.rect = words[a].rect;
		for (auto& char_: TextRecognizer::instance().recognize(words[a])) {
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
					for (auto& char_: TextRecognizer::instance().recognize(words[b])) {
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
}

}

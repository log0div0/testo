
#include "OCR.hpp"
#include <utf8.hpp>
#include <iostream>
#include <algorithm>
#include "TextDetector.hpp"
#include "TextRecognizer.hpp"
#include "TextColorPicker.hpp"

namespace nn {

bool Char::match(const stb::Image* image, const std::string& query) {
	for (auto& code: codes) {
		if (code == query) {
			return true;
		}
	}
	return false;
}

bool Char::match_foreground(const stb::Image* image, const std::string& color) {
	if (!color.size()) {
		return true;
	}
	if (!foreground.size()) {
		TextColorPicker::instance().run(image, *this);
	}
	return foreground == color;
}

bool Char::match_background(const stb::Image* image, const std::string& color) {
	if (!color.size()) {
		return true;
	}
	if (!background.size()) {
		TextColorPicker::instance().run(image, *this);
	}
	return background == color;
}

std::vector<TextLine> TextLine::match(const stb::Image* image, const std::string& text) {
	std::vector<std::string> query;
	for (auto& char_: utf8::split_to_chars(text)) {
		if (char_ == " ") {
			continue;
		}
		query.push_back(char_);
	}
	if (chars.size() < query.size()) {
		return {};
	}
	std::vector<TextLine> result;
	for (size_t i = 0; i < (chars.size() - query.size() + 1); ++i) {
		bool match = true;
		for (size_t j = 0; j < query.size(); ++j) {
			if (!chars[i + j].match(image, query[j])) {
				match = false;
				break;
			}
		}
		if (!match) {
			continue;
		}
		TextLine textline;
		textline.chars.push_back(chars[i]);
		textline.rect = chars[i].rect;
		for (size_t j = 1; j < query.size(); ++j) {
			textline.chars.push_back(chars[i + j]);
			textline.rect |= chars[i + j].rect;
		}
		result.push_back(textline);
	}
	return result;
}

bool TextLine::match_foreground(const stb::Image* image, const std::string& color) {
	size_t counter = 0;
	for (size_t i = 0; i < chars.size(); ++i) {
		if (chars[i].match_foreground(image, color)) {
			++counter;
		}
	}
	return counter >= (chars.size() / 2);
}

bool TextLine::match_background(const stb::Image* image, const std::string& color) {
	size_t counter = 0;
	for (size_t i = 0; i < chars.size(); ++i) {
		if (chars[i].match_background(image, color)) {
			++counter;
		}
	}
	return counter >= (chars.size() / 2);
}

Tensor Tensor::match(const stb::Image* image, const std::string& text) {
	Tensor result;
	for (auto& textline: textlines) {
		for (auto& new_textline: textline.match(image, text)) {
			result.textlines.push_back(new_textline);
		}
	}
	return result;
}

Tensor Tensor::match_foreground(const stb::Image* image, const std::string& color) {
	Tensor result;
	for (auto& textline: textlines) {
		if (textline.match_foreground(image, color)) {
			result.textlines.push_back(textline);
		}
	}
	return result;
}

Tensor Tensor::match_background(const stb::Image* image, const std::string& color) {
	Tensor result;
	for (auto& textline: textlines) {
		if (textline.match_background(image, color)) {
			result.textlines.push_back(textline);
		}
	}
	return result;
}

Tensor find_text(const stb::Image* image) {
	Tensor result;

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
		textline.rect = words[a].rect;
		for (auto& char_: TextRecognizer::instance().recognize(image, words[a])) {
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
					for (auto& char_: TextRecognizer::instance().recognize(image, words[b])) {
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
		result.textlines.push_back(textline);
	}

	std::sort(result.textlines.begin(), result.textlines.end(), [](const TextLine& a, const TextLine& b) {
		return a.rect.top < b.rect.top;
	});

	return result;
}

}


#include "TextLine.hpp"
#include <utf8.hpp>

namespace nn {

std::vector<TextLine> TextLine::match(const std::string& text) {
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
			if (!chars[i + j].match(query[j])) {
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

bool TextLine::match_foreground(const stb::Image<stb::RGB>* image, const std::string& color) {
	size_t counter = 0;
	for (size_t i = 0; i < chars.size(); ++i) {
		if (chars[i].match_foreground(image, color)) {
			++counter;
		}
	}
	return counter >= (chars.size() / 2);
}

bool TextLine::match_background(const stb::Image<stb::RGB>* image, const std::string& color) {
	size_t counter = 0;
	for (size_t i = 0; i < chars.size(); ++i) {
		if (chars[i].match_background(image, color)) {
			++counter;
		}
	}
	return counter >= (chars.size() / 2);
}

}

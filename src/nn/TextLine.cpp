
#include "TextLine.hpp"
#include <iostream>

namespace nn {

bool Char::match(const std::string& query) {
	for (auto& alternative: alternatives) {
		if (alternative == query) {
			return true;
		}
	}
	return false;
}

std::vector<Rect> TextLine::search(const std::vector<std::string>& query) {
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
		if (match) {
			Rect rect = chars[i].rect;
			for (size_t j = 1; j < query.size(); ++j) {
				rect |= chars[i + j].rect;
			}
			result.push_back(rect);
		}
	}
	return result;
}

}

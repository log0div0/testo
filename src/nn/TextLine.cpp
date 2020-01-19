
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

bool Word::match(const std::vector<std::string>& query) {
	if (chars.size() < query.size()) {
		return false;
	}
	for (size_t i = 0; i < (chars.size() - query.size() + 1); ++i) {
		bool match = true;
		for (size_t j = 0; j < query.size(); ++j) {
			if (!chars[i + j].match(query[j])) {
				match = false;
				break;
			}
		}
		if (match) {
			return true;
		}
	}
	return false;
}

bool Word::match_begin(const std::vector<std::string>& query) {
	if (chars.size() < query.size()) {
		return false;
	}
	for (size_t j = 0; j < query.size(); ++j) {
		if (!chars[j].match(query[j])) {
			return false;
		}
	}
	return true;
}

bool Word::match_end(const std::vector<std::string>& query) {
	if (chars.size() < query.size()) {
		return false;
	}
	size_t start = chars.size() - query.size();
	for (size_t j = 0; j < query.size(); ++j) {
		if (!chars[start + j].match(query[j])) {
			return false;
		}
	}
	return true;
}

}

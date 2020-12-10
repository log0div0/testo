
#pragma once

#include <vector>
#include <string>

namespace nn {

struct TextRecognizerCache {
	struct Prediction {
		bool maybe_blank = false;
		std::u32string codepoints;

		bool match(char32_t codepoint) const {
			for (char32_t cp: codepoints) {
				if (cp == codepoint) {
					return true;
				}
			}
			return false;
		}
	};

	std::vector<Prediction> predictions;

	int match(size_t x, const std::u32string& query) const;
	void print() const;
};

}

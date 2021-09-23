
#include "TextRecognizerCache.hpp"
#include <locale>
#include <codecvt>
#include <iostream>

namespace nn {

static std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;

int TextRecognizerCache::match(size_t x, const std::u32string& query) const {
	size_t y = 0;
	while (x < predictions.size() && (y < query.size())) {
		if (predictions[x].match(query[y])) {
			++y;
			++x;
			continue;
		}
		if ((y > 0) && predictions[x].match(query[y-1])) {
			++x;
			continue;
		}
		if (predictions[x].maybe_blank) {
			++x;
			continue;
		}
		break;
	}
	if (y == query.size()) {
		return x;
	} else {
		return -1;
	}
}

void TextRecognizerCache::print() const {
	for (size_t i = 0; i < predictions.size(); ++i) {
		auto prediction = predictions[i];
		std::cout << i << " ";
		if (prediction.maybe_blank) {
			std::cout << conv.to_bytes(U'•');
		}
		for (char32_t codepoint: prediction.codepoints) {
			std::cout << conv.to_bytes(codepoint);
		}
		std::cout << std::endl;
	}
}

}

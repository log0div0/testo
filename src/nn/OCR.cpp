
#include "OCR.hpp"
#include <iostream>
#include <algorithm>
#include "TextDetector.hpp"
#include "TextRecognizer.hpp"

namespace nn {

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

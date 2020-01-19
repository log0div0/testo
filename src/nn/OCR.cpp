
#include "OCR.hpp"
#include <utf8.hpp>
#include <iostream>
#include <algorithm>

namespace nn {

std::map<std::string, std::string> char_map = ([] {
	std::string common_str = "0123456789!?\"'#$%&@()[]{}<>+-*/\\.,:;^~=|_";
	std::string english_str = "abcdefghijklmnopqrstuvwxyz";
	std::string russian_str = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя";
	std::string ENGLISH_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	std::string RUSSIAN_str = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";
	std::string alphabet_str = common_str + ENGLISH_str + RUSSIAN_str;

	std::vector<std::string> english = utf8::split_to_chars(english_str);
	std::vector<std::string> russian = utf8::split_to_chars(russian_str);
	std::vector<std::string> ENGLISH = utf8::split_to_chars(ENGLISH_str);
	std::vector<std::string> RUSSIAN = utf8::split_to_chars(RUSSIAN_str);
	std::vector<std::string> alphabet = utf8::split_to_chars(alphabet_str);

	std::map<std::string, std::string> result;

	for (size_t i = 0; i < alphabet.size(); ++i) {
		char_map[alphabet[i]] = alphabet[i];
	}
	for (size_t i = 0; i < english.size(); ++i) {
		char_map[english[i]] = ENGLISH[i];
	}
	for (size_t i = 0; i < russian.size(); ++i) {
		char_map[russian[i]] = RUSSIAN[i];
	}
	char_map["«"] = char_map.at("\"");
	char_map["»"] = char_map.at("\"");

	return result;
})();

std::vector<Rect> OCRResult::search(const std::string& query) {
	std::vector<std::vector<std::string>> words;
	std::vector<std::string> word;
	for (auto& char_: utf8::split_to_chars(query)) {
		if (char_ == " ") {
			if (word.size()) {
				words.push_back(std::move(word));
			}
			continue;
		}
		word.push_back(char_map.at(char_));
	}
	if (word.size()) {
		words.push_back(std::move(word));
	}
	std::vector<Rect> result;
	for (auto& textline: textlines) {
			if (textline.words.size() < words.size()) {
				continue;
			}
			for (size_t i = 0; i < (textline.words.size() - words.size() + 1); ++i) {
				size_t first = 0;
				size_t last = words.size() - 1;
				if (textline.words[i + first].chars.size() < words.front().size()) {
					continue;
				}
				if (textline.words[i + last].chars.size() < words.back().size()) {
					continue;
				}
				bool ok = true;
				for (size_t j = 1; j < words.size() - 1; ++j) {
					if (textline.words[i + j].chars.size() != words[j].size()) {
						ok = false;
						break;
					}
				}
				if (!ok) {
					continue;
				}
				if (words.size() == 1) {
					if (textline.words[i].match(words[0])) {
						result.push_back(textline.words[i].rect);
					}
				} else {
					if (!textline.words[i + first].match_end(words.front())) {
						continue;
					}
					if (!textline.words[i + last].match_begin(words.back())) {
						continue;
					}
					ok = true;
					for (size_t j = 1; j < words.size() - 1; ++j) {
						if (!textline.words[i + j].match(words[j])) {
							ok = false;
							break;
						}
					}
					if (!ok) {
						continue;
					}
					Rect rect = textline.words[i].rect;
					for (size_t j = 1; j < words.size(); ++j) {
						rect |= textline.words[i + j].rect;
					}
					result.push_back(rect);
				}
			}
	}
	return result;
}

OCR& OCR::instance() {
	static OCR ocr;
	return ocr;
}

OCRResult OCR::run(const stb::Image& image) {
	OCRResult result;
	result.textlines = detector.detect(image);
	for (auto& textline: result.textlines) {
		for (auto& word: textline.words) {
			recognizer.recognize(image, word);
		}
	}
	return result;
}

}

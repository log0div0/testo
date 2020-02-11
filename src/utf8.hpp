
#pragma once

#include <string>
#include <vector>
#include <stdexcept>

namespace utf8 {

inline std::vector<std::string> split_to_chars(const std::string& str) {
	std::vector<std::string> result;
	size_t char_length = 0;
	size_t bytes_left = 0;
	for (size_t i = 0; i < str.size(); ++i) {
		char code_unit = str[i];
		if ((code_unit & 0b1100'0000) == 0b1000'0000) { // 10xxxxxx
			if (bytes_left == 0) {
				throw std::runtime_error("Unexpected 10xxxxxx byte in utf8 string");
			}
			--bytes_left;
		} else {
			if (bytes_left != 0) {
				throw std::runtime_error("Missing 10xxxxxx byte in utf8 string");
			}
			if ((code_unit & 0b1000'0000) == 0b0000'0000) { // 0xxxxxxx
				char_length = 1;
			} else if ((code_unit & 0b1110'0000) == 0b1100'0000) { // 110xxxxx
				char_length = 2;
			} else if ((code_unit & 0b1111'0000) == 0b1110'0000) { // 1110xxxx
				char_length = 3;
			} else if ((code_unit & 0b1111'1000) == 0b1111'0000) { // 11110xxx
				char_length = 4;
			} else {
				throw std::runtime_error("Invalid byte in utf8 string");
			}
			bytes_left = char_length - 1;
		}
		if (bytes_left == 0) {
			result.push_back(str.substr(i + 1 - char_length, char_length));
		}
	}
	if (bytes_left != 0) {
		throw std::runtime_error("Incomplete utf8 string");
	}
	return result;
}

}

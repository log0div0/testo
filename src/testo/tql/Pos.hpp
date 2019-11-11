
#pragma once

#include <string>
#include <iostream>
#include <stdexcept>
#include <memory>

namespace tql {

struct Pos {
	Pos() = default;
	Pos(std::shared_ptr<std::string> input): input(input) {}

	void advance(size_t shift = 1) {
		while (shift != 0) {
			if (offset == input->length()) {
				throw std::runtime_error("ADVANCE: Can't advance position over the end of the input");
			}
			if ((*input)[offset] == '\n') {
				line++;
				column = 1;
			} else {
				column++;
			}
			offset++;

			shift--;
		}
	}

	operator std::string() const {
		std::string result("");
		result += std::to_string(line) + ":" + std::to_string(column);
		return result;
	}

	operator size_t() const {
		return offset;
	}

	size_t offset = 0;
	uint32_t line = 1;
	uint32_t column = 1;

	std::shared_ptr<std::string> input;
};

}

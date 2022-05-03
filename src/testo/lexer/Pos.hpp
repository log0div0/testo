
#pragma once
#include <string>
#include <iostream>
#include <stdexcept>
#include "../Utils.hpp"

struct Pos {
	Pos() = default;
	Pos(const fs::path& file, std::shared_ptr<std::string> input): file(file), input(input) {
	}

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
		result += file.generic_string() + ":" + std::to_string(line) + ":" + std::to_string(column);
		return result;
	}

	bool operator==(const Pos& other) {
		return (offset == other.offset &&
			file == other.file);
	}

	static bool is_adjacent(Pos left, const Pos& right) {
		left.offset++;
		return left == right;
	}

	operator size_t() const {
		return offset;
	}

	size_t offset = 0;
	uint32_t line = 1;
	uint32_t column = 1;

	fs::path file; //should always be in canonical form

	std::shared_ptr<std::string> input; //we don't own the input, we just need to read it
};

inline std::ostream& operator<<(std::ostream& os, const Pos& pos)
{
	os << std::string(pos);
	return os;
}
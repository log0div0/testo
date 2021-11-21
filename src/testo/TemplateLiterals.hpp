
#pragma once

#include "Stack.hpp"
#include <vector>
#include <ostream>

namespace template_literals {

struct Pos {
	Pos() = default;
	Pos(const std::string& input): input(input) {}

	void advance(size_t shift = 1) {
		while (shift != 0) {
			if (offset == input.length()) {
				throw std::runtime_error("ADVANCE: Can't advance position over the end of the input");
			}
			if (input[offset] == '\n') {
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

	std::string input;
};

inline std::ostream& operator<<(std::ostream& os, const Pos& pos)
{
	os << std::string(pos);
	return os;
}


struct Token {
	enum category {
		eof,
		regular_string,
		param_ref,
		var_ref,

		//fake categories
		none, //not initialized
	};

	Token() {
		_type = none;
		_value = std::string("");
	}

	Token(category type, const std::string& value, Pos pos):
		_type(type), _value(value), _pos(pos) {}


	//we're doing getters to avoid occasional modifying of members
	category type() const {
		return _type;
	}

	std::string value() const {
		return _value;
	}

	Pos pos() const {
		return _pos;
	}

	operator bool() const {
		return _type != category::none;
	}

	operator std::string() const {
		return value();
	}


private:
	category _type;
	std::string _value;
	Pos _pos;
};


struct Resolver {
	Resolver() = default;
	Resolver(const std::string& input);

	std::string resolve(const std::shared_ptr<const StackNode>& stack) const;
	bool has_variables() const;
	bool can_resolve_variables() const;

private:
	std::string input;
	Pos current_pos;
	std::vector<Token> tokens;

	std::vector<Token> tokenize();
	Token ref();

	bool test_eof(size_t shift = 0) const { return ((current_pos + shift) >= input.length()); }
	bool test_escaped() const;
	bool test_ref() const;
	bool test_id(size_t shift = 0) const;
};

}

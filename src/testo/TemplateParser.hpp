
#pragma once
#include "Register.hpp"

namespace template_literals {

struct Pos {
	Pos() = default;
	Pos(const std::string& input): input(input) {}

	void advance(size_t shift = 1);

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



struct Parser {
	Parser() = default;

	std::string resolve(const std::string& input, Register& reg, std::shared_ptr<VmController> vmc = nullptr);

private:
	std::string input;
	Pos current_pos;

	std::vector<Token> tokenize();
	std::string resolve_var(const std::string& var, Register& reg, std::shared_ptr<VmController> vmc);

	bool test_eof(size_t shift = 0) const { return ((current_pos + shift) >= input.length()); }
	bool test_escaped() const;
	bool test_var_ref() const;
	bool test_id(size_t shift = 0) const;
	Token var_ref();
};


}

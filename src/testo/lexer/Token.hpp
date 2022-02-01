
#pragma once

#include "Pos.hpp"

struct Token {
	enum class category {
		none, //not initialized

		eof,
		newline,
		id,
		for_,
		test,
		machine,
		flash,
		network,
		param,
		macro,
		if_,
		else_,
		IN_,
		RANGE,
		break_,
		continue_,
		number,
		time_interval,
		size,
		quoted_string,
		triple_quoted_string,
		double_ampersand,	//&&
		double_vertical_bar,	//||
		exclamation_mark,	//!
		assign,
		dot,
		comma,
		plus,
		asterisk,
		lbrace,
		rbrace,
		lparen,
		rparen,
		lbracket,
		rbracket,
		semi,
		colon,
		include,

		//expressions
		DEFINED,

		//comparisons
		LESS,
		GREATER,
		EQUAL,
		STRLESS,
		STRGREATER,
		STREQUAL,

		//ops
		NOT,
		AND,
		OR,

		boolean, //true or false
	};

	Token() {
		_type = category::none;
		_value = std::string("");
	}

	Token(category type, const std::string& value, Pos begin, Pos end):
		_type(type), _value(value), _begin(begin), _end(end) {}


	//we're doing getters to avoid occasional modifying of members
	category type() const {
		return _type;
	}

	const std::string& value() const {
		return _value;
	}

	Pos begin() const {
		return _begin;
	}

	Pos end() const {
		return _end;
	}

	operator bool() const {
		return _type != category::none;
	}

	operator std::string() const {
		return value();
	}

	static std::string type_to_string(category type) {
		switch (type) {
		case category::eof:
			return "EOF";
		case category::newline:
			return "NEWLINE";
		case category::id:
			return "IDENTIFIER";
		case category::for_:
			return "FOR";
		case category::test:
			return "TEST";
		case category::machine:
			return "MACHINE";
		case category::flash:
			return "FLASH";
		case category::network:
			return "NETWORK";
		case category::param:
			return "PARAM";
		case category::macro:
			return "MACRO";
		case category::if_:
			return "IF";
		case category::else_:
			return "ELSE";
		case category::IN_:
			return "IN";
		case category::RANGE:
			return "RANGE";
		case category::number:
			return "NUMBER";
		case category::time_interval:
			return "TIME INTERVAL";
		case category::size:
			return "SIZE";
		case category::quoted_string:
			return "QUOTED STRING";
		case category::triple_quoted_string:
			return "TRIPLE QUOTED STRING";
		case category::assign:
			return "=";
		case category::exclamation_mark:
			return "!";
		case category::double_ampersand:
			return "&&";
		case category::double_vertical_bar:
			return "||";
		case category::dot:
			return ".";
		case category::comma:
			return ",";
		case category::plus:
			return "+";
		case category::asterisk:
			return "*";
		case category::lbrace:
			return "{";
		case category::rbrace:
			return "}";
		case category::lparen:
			return "(";
		case category::rparen:
			return ")";
		case category::lbracket:
			return "[";
		case category::rbracket:
			return "]";
		case category::semi:
			return ";";
		case category::colon:
			return ":";
		case category::DEFINED:
			return "DEFINED";
		case category::LESS:
			return "LESS";
		case category::GREATER:
			return "GREATER";
		case category::EQUAL:
			return "EQUAL";
		case category::STRLESS:
			return "STRLESS";
		case category::STRGREATER:
			return "STRGREATER";
		case category::STREQUAL:
			return "STREQUAL";
		case category::NOT:
			return "NOT";
		case category::AND:
			return "AND";
		case category::OR:
			return "OR";
		case category::none:
			return "NONE";
		case category::boolean:
			return "BOOLEAN CONSTANT";
		default:
			return "UNKNOWN TYPE";
		}
	}

private:
	category _type;
	std::string _value;
	Pos _begin;
	Pos _end;
};

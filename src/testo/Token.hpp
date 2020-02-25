
#pragma once

#include "Pos.hpp"

struct Token {
	enum category {
		eof,
		newline,
		id,
		abort,
		print,
		type_,
		wait,
		js,
		check,
		press,
		mouse,
		move,
		click,
		rclick,
		dclick,
		hold,
		release,
		lbtn,
		rbtn,
		plug,
		unplug,
		start,
		stop,
		shutdown,
		exec,
		copyto,
		copyfrom,
		timeout,
		for_,
		test,
		machine,
		flash,
		network,
		param,
		macro,
		dvd,
		if_,
		else_,
		in,
		break_,
		continue_,
		true_,
		false_,
		number,
		time_interval,
		size,
		quoted_string,
		triple_quoted_string,
		double_ampersand,	//&&
		double_vertical_bar,	//||
		exclamation_mark,	//!
		assign,
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
		double_dot,
		include,

		//expressions

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

		//fake categories
		none, //not initialized
		binary, //true or false
		program,
		key_spec,
		action_block,
		attr,
		attr_block,
		cmd,
		cmd_block
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

	static std::string type_to_string(category type) {
		switch (type) {
		case eof:
			return "EOF";
		case newline:
			return "NEWLINE";
		case id:
			return "IDENTIFIER";
		case abort:
			return "ACTION ABORT";
		case print:
			return "ACTION PRINT";
		case type_:
			return "ACTION TYPE";
		case wait:
			return "ACTION WAIT";
		case timeout:
			return "TIMEOUT";
		case js:
			return "JS";
		case press:
			return "ACTION PRESS";
		case mouse:
			return "MOUSE EVENT";
		case click:
			return "ACTION CLICK";
		case rclick:
			return "ACTION RCLICK";
		case dclick:
			return "ACTION DCLICK";
		case hold:
			return "ACTION HOLD";
		case release:
			return "ACTION RELEASE";
		case lbtn:
			return "LBTN";
		case rbtn:
			return "RBTN";
		case plug:
			return "ACTION PLUG";
		case unplug:
			return "ACTION UNPLUG";
		case start:
			return "ACTION START";
		case stop:
			return "ACTION STOP";
		case for_:
			return "FOR";
		case test:
			return "TEST";
		case machine:
			return "MACHINE";
		case flash:
			return "FLASH";
		case number:
			return "NUMBER";
		case time_interval:
			return "TIME INTERVAL";
		case size:
			return "SIZE";
		case quoted_string:
			return "QUOTED STRING";
		case triple_quoted_string:
			return "TRIPLE QUOTED STRING";
		case assign:
			return "=";
		case exclamation_mark:
			return "!";
		case double_ampersand:
			return "&&";
		case double_vertical_bar:
			return "||";
		case comma:
			return ",";
		case plus:
			return "+";
		case asterisk:
			return "*";
		case lbrace:
			return "{";
		case rbrace:
			return "}";
		case lparen:
			return "(";
		case rparen:
			return ")";
		case lbracket:
			return "[";
		case rbracket:
			return "]";
		case semi:
			return ";";
		case colon:
			return ":";
		case none:
			return "NONE";
		case program:
			return "PROGRAM";
		case key_spec:
			return "KEY SPECIFICATION";
		case action_block:
			return "ACTION BLOCK";
		case attr:
			return "MACHINE ATTRIBUTE";
		case attr_block:
			return "MACHINE ATTRIBUTE BLOCK";
		case cmd:
			return "COMMAND";
		case cmd_block:
			return "COMMAND BLOCK";
		case true_:
			return "TRUE";
		case false_:
			return "FALSE";
		case binary:
			return "BINARY CONSTANT";
		default:
			return "UNKNOWN TYPE";
		}
	}

private:
	category _type;
	std::string _value;
	Pos _pos;
};

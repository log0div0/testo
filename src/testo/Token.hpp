
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
		sleep,
		js,
		check,
		press,
		mouse,
		move,
		click,
		lclick,
		rclick,
		mclick,
		dclick,
		hold,
		release,
		lbtn,
		rbtn,
		mbtn,
		wheel,
		plug,
		unplug,
		start,
		stop,
		shutdown,
		exec,
		copyto,
		copyfrom,
		timeout,
		interval,
		for_,
		test,
		machine,
		flash,
		network,
		param,
		macro_action,
		dvd,
		if_,
		else_,
		IN_,
		RANGE,
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
		mouse_coordinates,
		mouse_selectable,
		mouse_additional_specifier,
		key_combination,
		string_token_union,
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

	Token(category type, const std::string& value, Pos begin, Pos end):
		_type(type), _value(value), _begin(begin), _end(end) {}


	//we're doing getters to avoid occasional modifying of members
	category type() const {
		return _type;
	}

	std::string value() const {
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
		case sleep:
			return "ACTION SLEEP";
		case js:
			return "JS";
		case check:
			return "CHECK";
		case press:
			return "ACTION PRESS";
		case timeout:
			return "TIMEOUT";
		case interval:
			return "INTERVAL";
		case mouse:
			return "MOUSE EVENT";
		case move:
			return "MOVE";
		case click:
			return "ACTION CLICK";
		case lclick:
			return "ACTION LCLICK";
		case rclick:
			return "ACTION RCLICK";
		case mclick:
			return "ACTION MCLICK";
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
		case mbtn:
			return "MBTN";
		case wheel:
			return "WHEEL";
		case plug:
			return "ACTION PLUG";
		case unplug:
			return "ACTION UNPLUG";
		case start:
			return "ACTION START";
		case stop:
			return "ACTION STOP";
		case shutdown:
			return "ACTION SHUTDOWN";
		case exec:
			return "ACTION EXEC";
		case copyto:
			return "ACTION COPYTO";
		case copyfrom:
			return "ACTION COPYFROM";
		case for_:
			return "FOR";
		case test:
			return "TEST";
		case machine:
			return "MACHINE";
		case flash:
			return "FLASH";
		case network:
			return "NETWORK";
		case param:
			return "PARAM";
		case macro_action:
			return "MACRO_ACTION";
		case dvd:
			return "DVD";
		case if_:
			return "IF";
		case else_:
			return "ELSE";
		case IN_:
			return "IN";
		case RANGE:
			return "RANGE";
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
		case dot:
			return ".";
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
		case mouse_selectable:
			return "MOUSE SELECTABLE";
		case mouse_additional_specifier:
			return "MOUSE ADDITIONAL SPECIFIER";
		case mouse_coordinates:
			return "MOUSE COORDINATES";
		case key_combination:
			return "KEY COMBINATION";
		case string_token_union:
			return "STRING TOKEN UNION";
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
	Pos _begin;
	Pos _end;
};

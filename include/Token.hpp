
#pragma once

#include <Pos.hpp>

struct Token {
	enum category {
		eof,
		newline,
		id,
		type_,
		wait,
		press,
		plug,
		unplug,
		start,
		stop,
		exec,
		set,
		copyto,
		for_,
		snapshot,
		test,
		machine,
		flash,
		dvd,
		number,
		time_interval,
		size,
		dbl_quoted_string,
		multiline_string,
		assign,
		comma,
		plus,
		asterisk,
		lbrace,
		rbrace,
		lparen,
		rparen,
		semi,
		colon,
		ifdef, //$ifdef
		ifndef, //$ifndef
		endif, //$ifndef
		else_, //$ifndef
		include, //$include

		//fake categories
		none, //not initialized
		program,
		key_spec,
		action_block,
		attr,
		attr_block,
		cmd,
		cmd_block,
		vm_state
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
		case type_:
			return "ACTION TYPE";
		case wait:
			return "ACTION WAIT";
		case press:
			return "ACTION PRESS";
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
		case snapshot:
			return "SNAPSHOT";
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
		case dbl_quoted_string:
			return "DOUBLE-QUOTED STRING";
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
		case vm_state:
			return "MACHINE STATE";
		default:
			return "UNKNOWN TYPE";
		}
	}

private:
	category _type;
	std::string _value;
	Pos _pos;
};

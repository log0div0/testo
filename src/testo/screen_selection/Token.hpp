
#pragma once

#include "Pos.hpp"

namespace screen_selection {

	struct Token {
		enum category {
			eof,
			number,
			dbl_quoted_string,
			equals,
			not_equals,
			comma,
			asterisk,
			lparen,
			rparen,

			id,
			select,
			from,
			where,

			not_,
			and_,
			or_,

			factor,
			expr,
			none
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
			case none:
				return "NONE";
			case id:
				return "identifier";
			case number:
				return "number";
			case dbl_quoted_string:
				return "double-quoted string";
			case equals:
				return "=";
			case not_equals:
				return "!=";
			case comma:
				return ",";
			case asterisk:
				return "*";
			case lparen:
				return "(";
			case rparen:
				return ")";
			case not_:
				return "not";
			case and_:
				return "and";
			case or_:
				return "or";
			default:
				return "UNKNOWN TOKEN TYPE";
			}
		}

	private:
		category _type;
		std::string _value;
		Pos _pos;
	};

}

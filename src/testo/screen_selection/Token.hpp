
#pragma once

#include "Pos.hpp"

namespace screen_selection {

	struct Token {
		enum category {
			eof,

			none,
			exec_plan
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
			case exec_plan:
				return "EXEC PLAN";
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

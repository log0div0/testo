
#pragma once

#include "Object.hpp"

namespace IR {

struct Param: Object<AST::Param> {
	static std::string type_name() { return "param"; }
	std::string value() const;
};

}

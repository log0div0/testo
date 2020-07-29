
#pragma once

#include "Object.hpp"

namespace IR {

struct Macro: Object<AST::Macro> {
	static std::string type_name() { return "macro"; }
};

}


#include "Param.hpp"
#include "../TemplateLiterals.hpp"

namespace IR {

std::string Param::value() const {
	return String(ast_node->value, stack).text();
}

}

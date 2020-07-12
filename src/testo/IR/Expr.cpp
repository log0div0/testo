
#include "Expr.hpp"
#include "Program.hpp"

namespace IR {

bool Defined::is_defined() const {
	return program->stack->is_defined(ast_node->var.value());
}


}
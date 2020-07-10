
#include "Param.hpp"
#include "../TemplateLiterals.hpp"

namespace IR {

std::string Param::value() const {
	return template_literals::Parser().resolve(ast_node->value->text(), stack);
}

}

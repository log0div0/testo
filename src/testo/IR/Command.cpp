
#include "Command.hpp"
#include "../TemplateLiterals.hpp"
#include "../Exceptions.hpp"

namespace IR {

std::string RegularCommand::entity() const {
	return Id(ast_node->entity, stack).value();
}

}
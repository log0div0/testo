
#include "Command.hpp"

namespace IR {

std::string RegularCommand::entity() const {
	return Id(ast_node->entity, stack).value();
}

}
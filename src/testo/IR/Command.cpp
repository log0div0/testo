
#include "Command.hpp"
#include "../TemplateLiterals.hpp"
#include "../Exceptions.hpp"

namespace IR {

std::string Command::entity() const {
	try {
		return template_literals::Parser().resolve(ast_node->entity->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->entity->begin(), ast_node->entity->text()));
	}
}

}
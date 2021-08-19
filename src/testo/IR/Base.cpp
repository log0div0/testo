
#include "Base.hpp"
#include "Program.hpp"
#include "../TemplateLiterals.hpp"
#include "../Exceptions.hpp"
#include "../Lexer.hpp"

namespace IR {

std::string StringTokenUnion::resolve() const {
	std::string result;

	if (ast_node->is_string()) {
		try {
			result = template_literals::Parser().resolve(ast_node->text(), stack);
			Lexer lex(".", result);

			try {
				if (lex.get_next_token().type() != ast_node->expected_token_type) {
					throw std::runtime_error("");
				}
			} catch(const std::exception& error) {
				throw std::runtime_error("Can't convert string value \"" + result +
					"\" to " + Token::type_to_string(ast_node->expected_token_type));
			}
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(ast_node->begin(), ast_node->text()));
		}

	} else {
		result = ast_node->token.value();
	}

	return result;
}

std::string OptionSeq::resolve(const std::string& name, const std::string& fallback_param) const {
	auto option = ast_node->get(name);
	if (option) {
		return StringTokenUnion(option->value, stack).resolve();
	} else {
		return program->stack->resolve_var(fallback_param);
	}
}

}

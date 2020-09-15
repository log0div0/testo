
#include "Action.hpp"
#include "Program.hpp"
#include "../TemplateLiterals.hpp"
#include "../Exceptions.hpp"
#include "../Lexer.hpp"

namespace IR {

std::string Abort::message() const {
	try {
		return template_literals::Parser().resolve(ast_node->message->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->message->begin(), ast_node->message->text()));
	}
}

std::string Print::message() const {
	try {
		return template_literals::Parser().resolve(ast_node->message->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->message->begin(), ast_node->message->text()));
	}
}

std::string Press::interval() const {
	if (ast_node->interval) {
		return StringTokenUnion(ast_node->interval, stack).resolve();
	} else {
		return program->stack->resolve_var("TESTO_PRESS_DEFAULT_INTERVAL");
	}
}

std::vector<std::string> Hold::buttons() const {
	return ast_node->combination->get_buttons();
}

std::vector<std::string> Release::buttons() const {
	if (ast_node->combination) {
		return ast_node->combination->get_buttons();
	}
	return {};
}

std::string Type::text() const {
	try {
		return template_literals::Parser().resolve(ast_node->text->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->text->begin(), ast_node->text->text()));
	}
}

std::string Type::interval() const {
	if (ast_node->interval) {
		return StringTokenUnion(ast_node->interval, stack).resolve();
	} else {
		return program->stack->resolve_var("TESTO_TYPE_DEFAULT_INTERVAL");
	}
}

std::string Wait::select_expr() const {
	try {
		return template_literals::Parser().resolve(std::string(*ast_node->select_expr), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->select_expr->begin(), std::string(*ast_node->select_expr)));
	}
}

std::string Wait::timeout() const {
	if (ast_node->timeout) {
		return StringTokenUnion(ast_node->timeout, stack).resolve();
	} else {
		return program->stack->resolve_var("TESTO_WAIT_DEFAULT_TIMEOUT");
	}
}

std::string Wait::interval() const {
	if (ast_node->interval) {
		return StringTokenUnion(ast_node->interval, stack).resolve();
	} else {
		return program->stack->resolve_var("TESTO_WAIT_DEFAULT_INTERVAL");
	}
}

std::string Sleep::timeout() const {
	return StringTokenUnion(ast_node->timeout, stack).resolve();
}

std::string MouseMoveClick::event_type() const {
	return ast_node->t.value();
}

std::string MouseCoordinates::x() const {
	return ast_node->dx.value();
}

std::string MouseCoordinates::y() const {
	return ast_node->dy.value();
}

std::string MouseSelectable::where_to_go() const {
	std::string result;
	if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(ast_node->selectable)) {
		result += "js selection \"";
		result += IR::SelectJS(p->selectable, stack).script();
		result += "\"";
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectText>>(ast_node->selectable)) {
		result += "\"";
		result += IR::SelectText(p->selectable, stack).text();
		result += "\"";
	} else {
		throw std::runtime_error("Where to go is unapplicable");
	}
	return result;
}

std::string MouseSelectable::timeout() const {
	if (ast_node->timeout) {
		return StringTokenUnion(ast_node->timeout, stack).resolve();
	} else {
		return program->stack->resolve_var("TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT");
	}
}

std::string SelectJS::script() const {
	try {
		return template_literals::Parser().resolve(ast_node->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->begin(), ast_node->text()));
	}
}

std::string SelectText::text() const {
	try {
		return template_literals::Parser().resolve(ast_node->text(), stack);
	} catch (const std::exception) {
		std::throw_with_nested(ResolveException(ast_node->begin(), ast_node->text()));
	}
}

std::string MouseHold::button() const {
	return ast_node->button.value();
}

bool Plug::is_on() const {
	return ast_node->is_on();
}

std::string Plug::entity_type() const {
	return ast_node->type.value();
}

std::string Plug::entity_name() const {
	if (ast_node->name) {
		return StringTokenUnion(ast_node->name, stack).resolve();
	}

	throw std::runtime_error("name is not defined");
}

fs::path Plug::dvd_path() const {
	fs::path path;
	try {
		path = template_literals::Parser().resolve(ast_node->path->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->path->begin(), ast_node->path->text()));
	}
	if (path.is_relative()) {
		path = ast_node->t.begin().file.parent_path() / path;
	}

	return path;
}

std::string Shutdown::timeout() const {
	if (ast_node->timeout) {
		return StringTokenUnion(ast_node->timeout, stack).resolve();
	} else {
		return "1m";
	}
}

std::string Exec::interpreter() const {
	return ast_node->process_token.value();

}

std::string Exec::timeout() const {
	if (ast_node->timeout) {
		return StringTokenUnion(ast_node->timeout, stack).resolve();
	} else {
		return program->stack->resolve_var("TESTO_EXEC_DEFAULT_TIMEOUT");
	}
}

std::string Exec::script() const {
	try {
		return template_literals::Parser().resolve(ast_node->commands->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->commands->begin(), ast_node->commands->text()));
	}
}

std::string Copy::from() const {
	fs::path from;
	try {
		from = template_literals::Parser().resolve(ast_node->from->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->from->begin(), ast_node->from->text()));
	}

	if (ast_node->is_to_guest()) {
		if (from.is_relative()) {
			from = ast_node->t.begin().file.parent_path() / from;
		}
	}

	return from.generic_string();
}

std::string Copy::to() const {
	fs::path to;
	try {
		to = template_literals::Parser().resolve(ast_node->to->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->to->begin(), ast_node->to->text()));
	}

	if (!ast_node->is_to_guest()) {
		if (to.is_relative()) {
			to = ast_node->t.begin().file.parent_path() / to;
		}
	}

	return to.generic_string();
}

std::string Copy::timeout() const {
	if (ast_node->timeout) {
		return StringTokenUnion(ast_node->timeout, stack).resolve();
	} else {
		return program->stack->resolve_var("TESTO_COPY_DEFAULT_TIMEOUT");
	}
}

std::string Check::timeout() const {
	if (ast_node->timeout) {
		return StringTokenUnion(ast_node->timeout, stack).resolve();
	} else {
		return program->stack->resolve_var("TESTO_CHECK_DEFAULT_TIMEOUT");
	}
}

std::string Check::interval() const {
	if (ast_node->interval) {
		return StringTokenUnion(ast_node->interval, stack).resolve();
	} else {
		return program->stack->resolve_var("TESTO_CHECK_DEFAULT_INTERVAL");
	}
}

std::string CycleControl::type() const {
	return ast_node->t.value();
}

std::string StringTokenUnion::resolve() const {
	std::string result;

	if (ast_node->string) {
		try {
			result = template_literals::Parser().resolve(ast_node->string->text(), stack);
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
			std::throw_with_nested(ResolveException(ast_node->string->begin(), ast_node->string->text()));
		}

	} else {
		result = ast_node->token.value();
	}

	return result;
}

}

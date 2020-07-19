
#include "Action.hpp"
#include "Program.hpp"
#include "../TemplateLiterals.hpp"
#include "../Lexer.hpp"

namespace IR {

std::string Abort::message() const {
	return template_literals::Parser().resolve(ast_node->message->text(), stack);
}

std::string Print::message() const {
	return template_literals::Parser().resolve(ast_node->message->text(), stack);
}

std::string Press::interval() const {
	if (ast_node->interval) {
		return ast_node->interval.value();
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
	return template_literals::Parser().resolve(ast_node->text->text(), stack);
}

std::string Type::interval() const {
	if (ast_node->interval) {
		return StringTokenUnion(ast_node->interval, stack).resolve();
	} else {
		return program->stack->resolve_var("TESTO_TYPE_DEFAULT_INTERVAL");
	}
}

std::string Wait::select_expr() const {
	return template_literals::Parser().resolve(std::string(*ast_node->select_expr), stack);
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
		return ast_node->timeout.value();
	} else {
		return program->stack->resolve_var("TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT");
	}
}

std::string SelectJS::script() const {
	return template_literals::Parser().resolve(ast_node->text(), stack);
}

std::string SelectText::text() const {
	return template_literals::Parser().resolve(ast_node->text(), stack);
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
	return ast_node->name_token.value();
}

fs::path Plug::dvd_path() const {
	fs::path path = template_literals::Parser().resolve(ast_node->path->text(), stack);
	if (path.is_relative()) {
		path = ast_node->t.begin().file.parent_path() / path;
	}

	return path;
}

std::string Shutdown::timeout() const {
	if (ast_node->timeout) {
		return ast_node->timeout.value();
	} else {
		return "1m";
	}
}

std::string Exec::interpreter() const {
	return ast_node->process_token.value();

}

std::string Exec::timeout() const {
	if (ast_node->time_interval) {
		return ast_node->time_interval.value();
	} else {
		return program->stack->resolve_var("TESTO_EXEC_DEFAULT_TIMEOUT");
	}
}

std::string Exec::script() const {
	return template_literals::Parser().resolve(ast_node->commands->text(), stack);
}

std::string Copy::from() const {
	fs::path from = template_literals::Parser().resolve(ast_node->from->text(), stack);

	if (ast_node->is_to_guest()) {
		if (from.is_relative()) {
			from = ast_node->t.begin().file.parent_path() / from;
		}
	}

	return from;
}

std::string Copy::to() const {
	fs::path to = template_literals::Parser().resolve(ast_node->to->text(), stack);

	if (!ast_node->is_to_guest()) {
		if (to.is_relative()) {
			to = ast_node->t.begin().file.parent_path() / to;
		}
	}

	return to;
}

std::string Copy::timeout() const {
	if (ast_node->time_interval) {
		return ast_node->time_interval.value();
	} else {
		return program->stack->resolve_var("TESTO_COPY_DEFAULT_TIMEOUT");
	}
}

std::string Check::timeout() const {
	if (ast_node->timeout) {
		return ast_node->timeout.value();
	} else {
		return program->stack->resolve_var("TESTO_CHECK_DEFAULT_TIMEOUT");
	}
}

std::string Check::interval() const {
	if (ast_node->interval) {
		return ast_node->interval.value();
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
		result = template_literals::Parser().resolve(ast_node->string->text(), stack);
		Lexer lex(".", result);

		try {
			if (lex.get_next_token().type() != ast_node->expected_token_type) {
				throw std::runtime_error("");
			}
		} catch(const std::exception& error) {
			throw std::runtime_error(std::string(ast_node->begin()) + ": Error: can't convert string value \"" + result +
				"\" to " + Token::type_to_string(ast_node->expected_token_type));
		}
	} else {
		result = ast_node->token.value();
	}

	return result;
}

}

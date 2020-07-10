
#include "Action.hpp"
#include "Program.hpp"
#include "../TemplateLiterals.hpp"

namespace IR {

std::string Press::interval() const {
	if (ast_node->interval) {
		return ast_node->interval.value();
	} else {
		return program->stack->resolve_var("TESTO_PRESS_DEFAULT_INTERVAL");
	}
}

std::string Type::text() const {
	return template_literals::Parser().resolve(ast_node->text->text(), stack);
}

std::string Type::interval() const {
	if (ast_node->interval) {
		return ast_node->interval.value();
	} else {
		return program->stack->resolve_var("TESTO_TYPE_DEFAULT_INTERVAL");
	}
}

std::string Wait::timeout() const {
	if (ast_node->timeout) {
		return ast_node->timeout.value();
	} else {
		return program->stack->resolve_var("TESTO_WAIT_DEFAULT_TIMEOUT");
	}
}

std::string Wait::interval() const {
	if (ast_node->interval) {
		return ast_node->interval.value();
	} else {
		return program->stack->resolve_var("TESTO_WAIT_DEFAULT_INTERVAL");
	}
}

std::string MouseSelectable::timeout() const {
	if (ast_node->timeout) {
		return ast_node->timeout.value();
	} else {
		return program->stack->resolve_var("TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT");
	}
}

std::string Exec::timeout() const {
	if (ast_node->time_interval) {
		return ast_node->time_interval.value();
	} else {
		return program->stack->resolve_var("TESTO_EXEC_DEFAULT_TIMEOUT");
	}
}

std::string Exec::text() const {
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

}

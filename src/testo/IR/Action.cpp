
#include "Action.hpp"
#include "Program.hpp"
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

int32_t KeySpec::times() const {
	if (ast_node->times) {
		return std::stoi(StringTokenUnion(ast_node->times, stack).resolve());
	} else {
		return 1;
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

bool MouseCoordinates::x_is_relative() const {
	return x().at(0) == '+' || x().at(0) == '-';
}

bool MouseCoordinates::y_is_relative() const {
	return y().at(0) == '+' || y().at(0) == '-';
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
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectImg>>(ast_node->selectable)) {
		result += "image \"";
		result += IR::SelectImg(p->selectable, stack).img_path().generic_string();
		result += "\"";
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectHomm3>>(ast_node->selectable)) {
		result += "HOMM3 object \"";
		result += IR::SelectHomm3(p->selectable, stack).id();
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

fs::path SelectImg::img_path() const {
	fs::path path;
	try {
		path = template_literals::Parser().resolve(ast_node->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->begin(), ast_node->text()));
	}
	if (path.is_relative()) {
		path = ast_node->t.begin().file.parent_path() / path;
	}

	return path;
}

std::string SelectHomm3::id() const {
	std::string id;
	try {
		id = template_literals::Parser().resolve(ast_node->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->begin(), ast_node->text()));
	}

	return id;
}


std::string SelectText::text() const {
	try {
		return template_literals::Parser().resolve(ast_node->text(), stack);
	} catch (const std::exception&) {
		std::throw_with_nested(ResolveException(ast_node->begin(), ast_node->text()));
	}
}

std::string MouseHold::button() const {
	return ast_node->button.value();
}

bool Plug::is_on() const {
	return ast_node->is_on();
}

std::string PlugFlash::name() const {
	return StringTokenUnion(ast_node->name, stack).resolve();
}

std::string PlugNIC::name() const {
	return StringTokenUnion(ast_node->name, stack).resolve();
}

std::string PlugLink::name() const {
	return StringTokenUnion(ast_node->name, stack).resolve();
}

std::string PlugHostDev::type() const {
	return ast_node->type.value();
}

std::string PlugHostDev::addr() const {
	try {
		return template_literals::Parser().resolve(ast_node->addr->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->addr->begin(), ast_node->addr->text()));
	}
}

fs::path PlugDVD::path() const {
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

std::string Screenshot::destination() const {
	fs::path dest;
	try {
		dest = template_literals::Parser().resolve(ast_node->destination->text(), stack);
		if (dest.is_relative()) {
			dest = ast_node->t.begin().file.parent_path() / dest;
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->destination->begin(), ast_node->destination->text()));
	}

	return dest.generic_string();
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

const std::shared_ptr<IR::Macro> MacroCall::get_macro() const {
	auto macro = program->get_macro_or_null(ast_node->name().value());
	if (!macro) {
		throw std::runtime_error(std::string(ast_node->begin()) + ": Error: unknown macro: " + ast_node->name().value());
	}
	return macro;
}

std::vector<std::pair<std::string, std::string>> MacroCall::args() const {
	std::vector<std::pair<std::string, std::string>> args;
	const std::shared_ptr<IR::Macro> macro = get_macro();

	for (size_t i = 0; i < ast_node->args.size(); ++i) {
		try {
			auto value = template_literals::Parser().resolve(ast_node->args[i]->text(), stack);
			args.push_back(std::make_pair(macro->ast_node->args[i]->name(), value));
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(ast_node->args[i]->begin(), ast_node->args[i]->text()));
		}
	}

	for (size_t i = ast_node->args.size(); i < macro->ast_node->args.size(); ++i) {
		try {
			auto value = template_literals::Parser().resolve(macro->ast_node->args[i]->default_value->text(), stack);
			args.push_back(std::make_pair(macro->ast_node->args[i]->name(), value));
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(macro->ast_node->args[i]->default_value->begin(), macro->ast_node->args[i]->default_value->text()));
		}
	}

	return args;
}

std::map<std::string, std::string> MacroCall::vars() const {
	std::map<std::string, std::string> vars;

	for (auto& kv: args()) {
		vars[kv.first] = kv.second;
	}

	return vars;
}

}

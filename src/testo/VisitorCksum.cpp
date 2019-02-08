
#include "VisitorCksum.hpp"
#include <algorithm>

using namespace AST;


uint64_t VisitorCksum::visit(std::shared_ptr<VmController> vm, std::shared_ptr<ActionBlock> action_block) {
	std::string result = visit_action_block(vm, action_block);
	std::hash<std::string> h;
	return h(result);
}

std::string VisitorCksum::visit_action_block(std::shared_ptr<VmController> vm, std::shared_ptr<ActionBlock> action_block) {
	std::string result("BLOCK");
	for (auto action: action_block->actions) {
		result += visit_action(vm, action);
	}
	return result;
}

std::string VisitorCksum::visit_action(std::shared_ptr<VmController> vm, std::shared_ptr<IAction> action) {
	if (auto p = std::dynamic_pointer_cast<Action<Type>>(action)) {
		return visit_type(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Wait>>(action)) {
		return visit_wait(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Press>>(action)) {
		return visit_press(p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Plug>>(action)) {
		return visit_plug(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Start>>(action)) {
		return "start";
	} else if (auto p = std::dynamic_pointer_cast<Action<Stop>>(action)) {
		return "stop";
	} else if (auto p = std::dynamic_pointer_cast<Action<Exec>>(action)) {
		return visit_exec(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Set>>(action)) {
		return visit_set(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<CopyTo>>(action)) {
		return visit_copyto(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<MacroCall>>(action)) {
		return visit_macro_call(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<IfClause>>(action)) {
		return visit_if_clause(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<ActionBlock>>(action)) {
		return visit_action_block(vm, p->action);
	} else {
		throw std::runtime_error("Unknown action");
	}
}

std::string VisitorCksum::visit_type(std::shared_ptr<VmController> vm, std::shared_ptr<Type> type) {
	std::string result("type");
	result += visit_word(vm, type->text_word);
	return result;
}

std::string VisitorCksum::visit_wait(std::shared_ptr<VmController> vm, std::shared_ptr<Wait> wait) {
	std::string result = "wait";
	if (wait->text_word) {
		result += visit_word(vm, wait->text_word);
	}

	if (wait->time_interval) {
		result += wait->time_interval.value();
	} else {
		result += "10s";
	}

	return result;
}

std::string VisitorCksum::visit_press(std::shared_ptr<Press> press) {
	std::string result("press");
	for (auto key_spec: press->keys) {
		result += visit_key_spec(key_spec);
	}
	return result;
}

std::string VisitorCksum::visit_key_spec(std::shared_ptr<KeySpec> key_spec) {
	std::string result("key_spec");
	result += key_spec->get_buttons_str();
	result += std::to_string(key_spec->get_times());
	return result;
}

std::string VisitorCksum::visit_plug(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	std::string result("plug");
	result += plug->is_on();
	result += plug->type.value();
	result += plug->name_token.value();
	if (plug->path) { //only for dvd
		result += visit_word(vm, plug->path);
	}
	return result;
}

std::string VisitorCksum::visit_exec(std::shared_ptr<VmController> vm, std::shared_ptr<Exec> exec) {
	std::string result("exec");
	if (!vm->has_key("login") || !vm->has_key("password")) {
		throw std::runtime_error(std::string(exec->begin()) + ": Error: This command requires login and password metadata for vm " + vm->name());
	}
	result += vm->get_metadata("login");
	result += vm->get_metadata("password");

	result += exec->process_token.value();
	result += visit_word(vm, exec->commands);
	return result;
}

std::string VisitorCksum::visit_set(std::shared_ptr<VmController> vm, std::shared_ptr<Set> set) {
	std::string result("set");
	for (auto assign: set->assignments) {
		result += assign->left.value();
		result += visit_word(vm, assign->right);
	}
	return result;
}

std::string VisitorCksum::visit_copyto(std::shared_ptr<VmController> vm, std::shared_ptr<CopyTo> copyto) {
	std::string result("copyto");
	if (!vm->has_key("login") || !vm->has_key("password")) {
		throw std::runtime_error(std::string(copyto->begin()) + ": Error: This command requires login and password metadata for vm " + vm->name());
	}
	result += vm->get_metadata("login");
	result += vm->get_metadata("password");

	result += visit_word(vm, copyto->from);
	result += visit_word(vm, copyto->to);
	return result;
}

std::string VisitorCksum::visit_macro_call(std::shared_ptr<VmController> vm, std::shared_ptr<MacroCall> macro_call) {
	return visit_action_block(vm, macro_call->macro->action_block->action);
}

std::string VisitorCksum::visit_if_clause(std::shared_ptr<VmController> vm, std::shared_ptr<IfClause> if_clause) {
	std::string result("if");
	result += visit_expr(vm, if_clause->expr);

	result += visit_action(vm, if_clause->if_action);

	if (if_clause->has_else()) {
		result += visit_action(vm, if_clause->else_action);
	}

	return result;
}

std::string VisitorCksum::visit_expr(std::shared_ptr<VmController> vm, std::shared_ptr<IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<Expr<BinOp>>(expr)) {
		return visit_binop(vm, p->expr);
	} else if (auto p = std::dynamic_pointer_cast<Expr<IFactor>>(expr)) {
		return visit_factor(vm, p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

std::string VisitorCksum::visit_binop(std::shared_ptr<VmController> vm, std::shared_ptr<BinOp> binop) {
	std::string result("binop");
	result += visit_expr(vm, binop->left);
	result += visit_expr(vm, binop->right);

	return result;
}

std::string VisitorCksum::visit_factor(std::shared_ptr<VmController> vm, std::shared_ptr<IFactor> factor) {
	std::string result("factor");
	if (auto p = std::dynamic_pointer_cast<Factor<Word>>(factor)) {
		result += p->is_negated();
		result += visit_word(vm, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<Comparison>>(factor)) {
		result += p->is_negated();
		result += visit_comparison(vm, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<IExpr>>(factor)) {
		result += p->is_negated();
		result += visit_expr(vm, p->factor);
	} else {
		throw std::runtime_error("Unknown factor type");
	}

	return result;
}

std::string VisitorCksum::resolve_var(std::shared_ptr<VmController> vm, const std::string& var) {
	//Resolving order
	//1) metadata
	//2) reg (todo)
	//3) env var

	if (vm->has_key(var)) {
		return vm->get_metadata(var);
	}

	auto env_value = std::getenv(var.c_str());

	if (env_value == nullptr) {
		return "";
	}
	return env_value;
}

std::string VisitorCksum::visit_word(std::shared_ptr<VmController> vm, std::shared_ptr<Word> word) {
	std::string result;

	for (auto part: word->parts) {
		if (part.type() == Token::category::dbl_quoted_string) {
			result += part.value().substr(1, part.value().length() - 2);
		} else if (part.type() == Token::category::var_ref) {
			result += resolve_var(vm, part.value().substr(1, part.value().length() - 1));
		} else if (part.type() == Token::category::multiline_string) {
			result += part.value().substr(3, part.value().length() - 6);
		} else {
			throw std::runtime_error("Unknown word type");
		}
	}

	return result;
}

std::string VisitorCksum::visit_comparison(std::shared_ptr<VmController> vm, std::shared_ptr<Comparison> comparison) {
	std::string result("comparison");
	result += visit_word(vm, comparison->left);
	result += visit_word(vm, comparison->right);
	result += comparison->op();
	return result;
}



#include "VisitorCksum.hpp"
#include <algorithm>

using namespace AST;

//TODO: Snapshot cksum must depend on name and parent cksum

uint64_t VisitorCksum::visit(std::shared_ptr<VmController> vm, std::shared_ptr<Snapshot> snapshot) {
	std::string result = snapshot->name.value();
	if (snapshot->parent) {
		result += snapshot->parent->name.value();
	}
	result += visit_action_block(vm, snapshot->action_block->action);
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
	} else if (auto p = std::dynamic_pointer_cast<Action<Copy>>(action)) {
		return visit_copy(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<MacroCall>>(action)) {
		return visit_macro_call(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<IfClause>>(action)) {
		return visit_if_clause(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<ForClause>>(action)) {
		return visit_for_clause(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<CycleControl>>(action)) {
		return p->action->t.value();
	} else if (auto p = std::dynamic_pointer_cast<Action<ActionBlock>>(action)) {
		return visit_action_block(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Empty>>(action)) {
		return "";
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

	result += "(";
	for (auto param: wait->params) {
		auto value = visit_word(vm, param->right);
		result += param->left.value() + "=" + value;
	}
	result += ")";

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
	result += std::to_string(plug->is_on());
	result += plug->type.value();
	result += plug->name_token.value();
	if (plug->path) { //only for dvd
		fs::path path = visit_word(vm, plug->path);
		if (path.is_relative()) {
			path = plug->t.pos().file.parent_path() / path;
		}
		result += path.generic_string();
		//add signature for dvd file
		result += file_signature(path);
	}

	if (plug->type.value() == "flash") {
		auto fd = reg.fds.find(plug->name_token.value())->second; //should always be found
		result += fd->cksum();
	}
	return result;
}

std::string VisitorCksum::visit_exec(std::shared_ptr<VmController> vm, std::shared_ptr<Exec> exec) {
	std::string result("exec");

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

std::string VisitorCksum::visit_copy(std::shared_ptr<VmController> vm, std::shared_ptr<Copy> copy) {
	std::string result(copy->t.value());

	fs::path from = visit_word(vm, copy->from);

	if (from.is_relative()) {
		from = copy->t.pos().file.parent_path() / from;
	}

	result += from.generic_string();

	if (copy->is_to_guest()) {
		if (!fs::exists(from)) {
			throw std::runtime_error("Specified path doesn't exist: " + fs::path(from).generic_string());
		}

		//now we should take care of last modify date of every file and folder in the folder
		if (fs::is_regular_file(from)) {
			result += file_signature(from);
		} else if (fs::is_directory(from)) {
			result += directory_signature(from);
		} else {
			throw std::runtime_error("Unknown type of file: " + fs::path(from).generic_string());
		}
	} //TODO: I wonder what it must be for the from copy

	fs::path to = visit_word(vm, copy->to);

	if (to.is_relative()) {
		to = copy->t.pos().file.parent_path() / to;
	}

	result += to.generic_string();

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
		result += "else";
		result += visit_action(vm, if_clause->else_action);
	}

	return result;
}

std::string VisitorCksum::visit_for_clause(std::shared_ptr<VmController> vm, std::shared_ptr<ForClause> for_clause) {
	std::string result("for");
	//we should drop the counter from cksum
	result += for_clause->start_.value();
	result += "..";
	result += for_clause->finish_.value();
	result += visit_action(vm, for_clause->cycle_body);

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
		result += std::to_string(p->is_negated());
		result += visit_word(vm, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<Comparison>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_comparison(vm, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<Check>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_check(vm, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<IExpr>>(factor)) {
		result += std::to_string(p->is_negated());
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
	result += comparison->op().value();
	return result;
}

std::string VisitorCksum::visit_check(std::shared_ptr<VmController> vm, std::shared_ptr<Check> check) {
	std::string result = "check";
	result += visit_word(vm, check->text_word);

	result += "(";
	for (auto param: check->params) {
		auto value = visit_word(vm, param->right);
		result += param->left.value() + "=" + value;
	}
	result += ")";

	return result;
}

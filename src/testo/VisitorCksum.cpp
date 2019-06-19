
#include "VisitorCksum.hpp"
#include <algorithm>

using namespace AST;

uint64_t VisitorCksum::visit(std::shared_ptr<Test> test) {
	std::string result = test->name.value();

	for (auto parent: test->parents) {
		result += parent->name.value();
	}

	for (auto cmd: test->cmd_block->commands) {
		result += visit_cmd(cmd);
	}
	std::hash<std::string> h;
	return h(result);
}

std::string VisitorCksum::visit_cmd(std::shared_ptr<Cmd> cmd) {
	std::string result;

	for (auto vm_token: cmd->vms) {
		result += vm_token.value();
		auto vmc = reg.vmcs.find(vm_token);
		result += visit_action(vmc->second, cmd->action);
	}
	return result;
}

std::string VisitorCksum::visit_action_block(std::shared_ptr<VmController> vmc, std::shared_ptr<ActionBlock> action_block) {
	std::string result("BLOCK");
	for (auto action: action_block->actions) {
		result += visit_action(vmc, action);
	}
	return result;
}

std::string VisitorCksum::visit_action(std::shared_ptr<VmController> vmc, std::shared_ptr<IAction> action) {
	if (auto p = std::dynamic_pointer_cast<Action<Type>>(action)) {
		return visit_type(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Wait>>(action)) {
		return visit_wait(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Press>>(action)) {
		return visit_press(p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Plug>>(action)) {
		return visit_plug(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Shutdown>>(action)) {
		return visit_shutdown(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Start>>(action)) {
		return "start";
	} else if (auto p = std::dynamic_pointer_cast<Action<Stop>>(action)) {
		return "stop";
	} else if (auto p = std::dynamic_pointer_cast<Action<Exec>>(action)) {
		return visit_exec(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Copy>>(action)) {
		return visit_copy(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<MacroCall>>(action)) {
		return visit_macro_call(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<IfClause>>(action)) {
		return visit_if_clause(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<ForClause>>(action)) {
		return visit_for_clause(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<CycleControl>>(action)) {
		return p->action->t.value();
	} else if (auto p = std::dynamic_pointer_cast<Action<ActionBlock>>(action)) {
		return visit_action_block(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Empty>>(action)) {
		return "";
	} else {
		throw std::runtime_error("Unknown action");
	}
}

std::string VisitorCksum::visit_type(std::shared_ptr<VmController> vmc, std::shared_ptr<Type> type) {
	std::string result("type");
	result += visit_word(vmc, type->text_word);
	return result;
}

std::string VisitorCksum::visit_wait(std::shared_ptr<VmController> vmc, std::shared_ptr<Wait> wait) {
	std::string result = "wait";
	if (wait->text_word) {
		result += visit_word(vmc, wait->text_word);
	}

	result += "(";
	for (auto param: wait->params) {
		auto value = visit_word(vmc, param->right);
		result += param->left.value() + "=" + value;
	}
	result += ")";

	if (wait->time_interval) {
		result += wait->time_interval.value();
	} else {
		result += "1m";
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

std::string VisitorCksum::visit_plug(std::shared_ptr<VmController> vmc, std::shared_ptr<Plug> plug) {
	std::string result("plug");
	result += std::to_string(plug->is_on());
	result += plug->type.value();
	result += plug->name_token.value();
	if (plug->path) { //only for dvd
		fs::path path = visit_word(vmc, plug->path);
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

std::string VisitorCksum::visit_shutdown(std::shared_ptr<VmController>, std::shared_ptr<Shutdown> shutdown) {
	std::string result("shutdown");
	if (shutdown->time_interval) {
		result += shutdown->time_interval.value();
	} else {
		result += "1m";
	}
	return result;
}

std::string VisitorCksum::visit_exec(std::shared_ptr<VmController> vmc, std::shared_ptr<Exec> exec) {
	std::string result("exec");

	result += exec->process_token.value();
	result += visit_word(vmc, exec->commands);

	if (exec->time_interval) {
		result += exec->time_interval.value();
	} else {
		result += "600s";
	}

	return result;
}

std::string VisitorCksum::visit_copy(std::shared_ptr<VmController> vmc, std::shared_ptr<Copy> copy) {
	std::string result(copy->t.value());

	fs::path from = visit_word(vmc, copy->from);

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

	fs::path to = visit_word(vmc, copy->to);

	if (to.is_relative()) {
		to = copy->t.pos().file.parent_path() / to;
	}

	result += to.generic_string();

	if (copy->time_interval) {
		result += copy->time_interval.value();
	} else {
		result += "600s";
	}

	return result;
}

std::string VisitorCksum::visit_macro_call(std::shared_ptr<VmController> vmc, std::shared_ptr<MacroCall> macro_call) {
	return visit_action_block(vmc, macro_call->macro->action_block->action);
}

std::string VisitorCksum::visit_if_clause(std::shared_ptr<VmController> vmc, std::shared_ptr<IfClause> if_clause) {
	std::string result("if");
	result += visit_expr(vmc, if_clause->expr);

	result += visit_action(vmc, if_clause->if_action);

	if (if_clause->has_else()) {
		result += "else";
		result += visit_action(vmc, if_clause->else_action);
	}

	return result;
}

std::string VisitorCksum::visit_for_clause(std::shared_ptr<VmController> vmc, std::shared_ptr<ForClause> for_clause) {
	std::string result("for");
	//we should drop the counter from cksum
	result += for_clause->start_.value();
	result += "..";
	result += for_clause->finish_.value();
	result += visit_action(vmc, for_clause->cycle_body);

	return result;
}

std::string VisitorCksum::visit_expr(std::shared_ptr<VmController> vmc, std::shared_ptr<IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<Expr<BinOp>>(expr)) {
		return visit_binop(vmc, p->expr);
	} else if (auto p = std::dynamic_pointer_cast<Expr<IFactor>>(expr)) {
		return visit_factor(vmc, p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

std::string VisitorCksum::visit_binop(std::shared_ptr<VmController> vmc, std::shared_ptr<BinOp> binop) {
	std::string result("binop");
	result += visit_expr(vmc, binop->left);
	result += visit_expr(vmc, binop->right);

	return result;
}

std::string VisitorCksum::visit_factor(std::shared_ptr<VmController> vmc, std::shared_ptr<IFactor> factor) {
	std::string result("factor");
	if (auto p = std::dynamic_pointer_cast<Factor<Word>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_word(vmc, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<Comparison>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_comparison(vmc, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<Check>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_check(vmc, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<IExpr>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_expr(vmc, p->factor);
	} else {
		throw std::runtime_error("Unknown factor type");
	}

	return result;
}

std::string VisitorCksum::resolve_var(std::shared_ptr<VmController> vmc, const std::string& var) {
	//Resolving order
	//1) metadata
	//2) reg (todo)
	//3) env var

	if (vmc->vm->is_defined() && vmc->vm->has_key(var)) {
		return vmc->get_metadata(var);
	}

	auto env_value = std::getenv(var.c_str());

	if (env_value == nullptr) {
		return "";
	}
	return env_value;
}

std::string VisitorCksum::visit_word(std::shared_ptr<VmController> vmc, std::shared_ptr<Word> word) {
	std::string result;

	for (auto part: word->parts) {
		if (part.type() == Token::category::dbl_quoted_string) {
			result += part.value().substr(1, part.value().length() - 2);
		} else if (part.type() == Token::category::var_ref) {
			result += resolve_var(vmc, part.value().substr(1, part.value().length() - 1));
		} else if (part.type() == Token::category::multiline_string) {
			result += part.value().substr(3, part.value().length() - 6);
		} else {
			throw std::runtime_error("Unknown word type");
		}
	}

	return result;
}

std::string VisitorCksum::visit_comparison(std::shared_ptr<VmController> vmc, std::shared_ptr<Comparison> comparison) {
	std::string result("comparison");
	result += visit_word(vmc, comparison->left);
	result += visit_word(vmc, comparison->right);
	result += comparison->op().value();
	return result;
}

std::string VisitorCksum::visit_check(std::shared_ptr<VmController> vmc, std::shared_ptr<Check> check) {
	std::string result = "check";
	result += visit_word(vmc, check->text_word);

	result += "(";
	for (auto param: check->params) {
		auto value = visit_word(vmc, param->right);
		result += param->left.value() + "=" + value;
	}
	result += ")";

	return result;
}

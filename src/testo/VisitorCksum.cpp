
#include "VisitorCksum.hpp"
#include "backends/Environment.hpp"
#include "coro/Finally.h"
#include <algorithm>

uint64_t VisitorCksum::visit(std::shared_ptr<AST::Test> test) {
	std::string result = test->name.value();

	for (auto parent: test->parents) {
		result += parent->name.value();
	}

	result += test->snapshots_needed;

	for (auto cmd: test->cmd_block->commands) {
		result += visit_cmd(cmd);
	}

	std::hash<std::string> h;
	return h(result);
}

std::string VisitorCksum::visit_cmd(std::shared_ptr<AST::Cmd> cmd) {
	std::string result;

	for (auto vm_token: cmd->vms) {
		result += vm_token.value();
		auto vmc = reg->vmcs.find(vm_token);
		result += visit_action(vmc->second, cmd->action);
	}
	return result;
}

std::string VisitorCksum::visit_action_block(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::ActionBlock> action_block) {
	std::string result("BLOCK");
	for (auto action: action_block->actions) {
		result += visit_action(vmc, action);
	}
	return result;
}

std::string VisitorCksum::visit_action(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Abort>>(action)) {
		return visit_abort(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Print>>(action)) {
		return visit_print(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Type>>(action)) {
		return visit_type(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Wait>>(action)) {
		return visit_wait(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		return visit_sleep(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Press>>(action)) {
		return visit_press(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Hold>>(action)) {
		return std::string(*p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Release>>(action)) {
		return std::string(*p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Mouse>>(action)) {
		return visit_mouse(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Plug>>(action)) {
		return visit_plug(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Shutdown>>(action)) {
		return visit_shutdown(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Start>>(action)) {
		return "start";
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Stop>>(action)) {
		return "stop";
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Exec>>(action)) {
		return visit_exec(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Copy>>(action)) {
		return visit_copy(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroActionCall>>(action)) {
		return visit_macro_action_call(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		return visit_if_clause(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		return visit_for_clause(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::CycleControl>>(action)) {
		return p->action->t.value();
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		return visit_action_block(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Empty>>(action)) {
		return "";
	} else {
		throw std::runtime_error("Unknown action");
	}
}

std::string VisitorCksum::visit_abort(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Abort> abort) {
	std::string result("abort");
	result += template_parser.resolve(abort->message->text(), reg);
	return result;
}

std::string VisitorCksum::visit_print(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Print> print) {
	std::string result("print");
	result += template_parser.resolve(print->message->text(), reg);
	return result;
}

std::string VisitorCksum::visit_press(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Press> press) {
	std::string result = std::string(*press);
	if (!press->interval) {
		auto press_interval_found = reg->params.find("TESTO_PRESS_DEFAULT_INTERVAL");
		result += (press_interval_found != reg->params.end()) ? press_interval_found->second : "30ms";
	}
	return result;
}

std::string VisitorCksum::visit_type(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Type> type) {
	std::string result("type");
	result += template_parser.resolve(type->text->text(), reg);
	if (type->interval) {
		result += type->interval.value();
	} else {
		auto type_interval_found = reg->params.find("TESTO_TYPE_DEFAULT_INTERVAL");
		result += (type_interval_found != reg->params.end()) ? type_interval_found->second : "30ms";
	}
	return result;
}

std::string VisitorCksum::visit_wait(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Wait> wait) {
	std::string result = "wait";
	result += template_parser.resolve(std::string(*wait->select_expr), reg);

	if (wait->timeout) {
		result += template_parser.resolve(wait->timeout->text(), reg);
	} else {
		auto wait_timeout_found = reg->params.find("TESTO_WAIT_DEFAULT_TIMEOUT");
		result += (wait_timeout_found != reg->params.end()) ? wait_timeout_found->second : "1m";
	}

	if (wait->interval) {
		result += template_parser.resolve(wait->interval->text(), reg);
	} else {
		auto wait_interval_found = reg->params.find("TESTO_WAIT_DEFAULT_INTERVAL");
		result += (wait_interval_found != reg->params.end()) ? wait_interval_found->second : "1s";
	}

	return result;
}

std::string VisitorCksum::visit_sleep(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Sleep> sleep) {
	std::string result = "sleep";
	result += template_parser.resolve(sleep->timeout->text(), reg);
	
	return result;
}

std::string VisitorCksum::visit_mouse(std::shared_ptr<AST::Mouse> mouse) {
	std::string result = "mouse";

	if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseMoveClick>>(mouse->event)) {
		result += visit_mouse_move_click(p->event);
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseHold>>(mouse->event)) {
		result += std::string(*(p->event));
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseRelease>>(mouse->event)) {
		result += std::string(*(p->event));
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseWheel>>(mouse->event)) {
		result += std::string(*(p->event));
	} else {
		throw std::runtime_error("Unknown mouse action");
	}

	return result;
}

std::string VisitorCksum::visit_mouse_move_click(std::shared_ptr<AST::MouseMoveClick> mouse_move_click) {
	std::string result = mouse_move_click->t.value();
	if (mouse_move_click->object) {
		result += visit_mouse_move_target(mouse_move_click->object);
	}

	return result;
}


std::string VisitorCksum::visit_mouse_selectable(std::shared_ptr<AST::MouseSelectable> mouse_selectable) {
	std::string result = template_parser.resolve(mouse_selectable->text(), reg);

	for (auto specifier: mouse_selectable->specifiers) {
		result += std::string(*specifier);
	}

	if (mouse_selectable->timeout) {
		result += mouse_selectable->timeout.value();
	} else {
		auto mouse_move_click_timeout_found = reg->params.find("TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT");
		result += (mouse_move_click_timeout_found != reg->params.end()) ? mouse_move_click_timeout_found->second : "1m";
	}

	return result;
}

std::string VisitorCksum::visit_mouse_move_target(std::shared_ptr<AST::IMouseMoveTarget> target) {
	std::string result;
	if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseCoordinates>>(target)) {
		result = std::string(*p->target);
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseSelectable>>(target)) {
		result = visit_mouse_selectable(p->target);
	} else {
		throw std::runtime_error("Unknown mouse even object");
	}
	return result;
}


std::string VisitorCksum::visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec) {
	std::string result("key_spec");
	result += std::string(*key_spec->combination);
	result += std::to_string(key_spec->get_times());
	return result;
}

std::string VisitorCksum::visit_plug(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
	std::string result("plug");
	result += std::to_string(plug->is_on());
	result += plug->type.value();
	result += plug->name_token.value();
	if (plug->path) { //only for dvd
		fs::path path = template_parser.resolve(plug->path->text(), reg);
		if (path.is_relative()) {
			path = plug->t.begin().file.parent_path() / path;
		}
		//add signature for dvd file
		result += file_signature(path, env->content_cksum_maxsize());
	}

	return result;
}

std::string VisitorCksum::visit_shutdown(std::shared_ptr<VmController>, std::shared_ptr<AST::Shutdown> shutdown) {
	std::string result("shutdown");
	if (shutdown->timeout) {
		result += template_parser.resolve(shutdown->timeout->text(), reg);
	} else {
		result += "1m";
	}
	return result;
}

std::string VisitorCksum::visit_exec(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Exec> exec) {
	std::string result("exec");

	result += exec->process_token.value();
	result += template_parser.resolve(exec->commands->text(), reg);

	if (exec->time_interval) {
		result += exec->time_interval.value();
	} else {
		auto exec_default_timeout_found = reg->params.find("TESTO_EXEC_DEFAULT_TIMEOUT");
		result += (exec_default_timeout_found != reg->params.end()) ? exec_default_timeout_found->second : "10m";
	}

	return result;
}

std::string VisitorCksum::visit_copy(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Copy> copy) {
	std::string result(copy->t.value());

	fs::path from = template_parser.resolve(copy->from->text(), reg);

	if (from.is_relative()) {
		from = copy->t.begin().file.parent_path() / from;
	}

	result += from.generic_string();

	if (copy->is_to_guest()) {
		if (!fs::exists(from)) {
			throw std::runtime_error("Specified path doesn't exist: " + fs::path(from).generic_string());
		}

		//now we should take care of last modify date of every file and folder in the folder
		if (fs::is_regular_file(from)) {
			result += file_signature(from, env->content_cksum_maxsize());
		} else if (fs::is_directory(from)) {
			result += directory_signature(from, env->content_cksum_maxsize());
		} else {
			throw std::runtime_error("Unknown type of file: " + fs::path(from).generic_string());
		}
	} //TODO: I wonder what it must be for the from copy

	fs::path to = template_parser.resolve(copy->to->text(), reg);

	if (to.is_relative()) {
		to = copy->t.begin().file.parent_path() / to;
	}

	result += to.generic_string();

	if (copy->timeout) {
		result += template_parser.resolve(copy->timeout->text(), reg);
	} else {
		auto copy_default_timeout_found = reg->params.find("TESTO_COPY_DEFAULT_TIMEOUT");
		result += (copy_default_timeout_found != reg->params.end()) ? copy_default_timeout_found->second : "10m";
	}

	return result;
}

std::string VisitorCksum::visit_macro_action_call(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MacroActionCall> macro_action_call) {
	StackEntry new_ctx(true);

	for (size_t i = 0; i < macro_action_call->args.size(); ++i) {
		auto value = template_parser.resolve(macro_action_call->args[i]->text(), reg);
		new_ctx.define(macro_action_call->macro_action->args[i]->name(), value);
	}

	for (size_t i = macro_action_call->args.size(); i < macro_action_call->macro_action->args.size(); ++i) {
		auto value = template_parser.resolve(macro_action_call->macro_action->args[i]->default_value->text(), reg);
		new_ctx.define(macro_action_call->macro_action->args[i]->name(), value);
	}

	reg->local_vars.push_back(new_ctx);
	coro::Finally finally([&] {
		reg->local_vars.pop_back();
	});

	return visit_action_block(vmc, macro_action_call->macro_action->action_block->action);
}

std::string VisitorCksum::visit_if_clause(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IfClause> if_clause) {
	std::string result("if");
	result += visit_expr(vmc, if_clause->expr);

	result += visit_action(vmc, if_clause->if_action);

	if (if_clause->has_else()) {
		result += "else";
		result += visit_action(vmc, if_clause->else_action);
	}

	return result;
}

std::string VisitorCksum::visit_range(std::shared_ptr<AST::Range> range) {
	std::string result = "RANGE";

	result += template_parser.resolve(range->r1->text(), reg);

	if (range->r2) {
		result += " ";
		result += template_parser.resolve(range->r2->text(), reg);
	}

	return result;
}

std::string VisitorCksum::visit_for_clause(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::ForClause> for_clause) {
	std::string result("for");
	//we should drop the counter from cksum

	if (auto p = std::dynamic_pointer_cast<AST::CounterList<AST::Range>>(for_clause->counter_list)) {
		result += visit_range(p->counter_list);
	} else {
		throw std::runtime_error("Unknown counter list");
	}

	result += visit_action(vmc, for_clause->cycle_body);
	if (for_clause->else_token) {
		result += "else";
		result += visit_action(vmc, for_clause->else_action);
	}
	return result;
}

std::string VisitorCksum::visit_expr(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::BinOp>>(expr)) {
		return visit_binop(vmc, p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::IFactor>>(expr)) {
		return visit_factor(vmc, p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

std::string VisitorCksum::visit_binop(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::BinOp> binop) {
	std::string result("binop");
	result += visit_expr(vmc, binop->left);
	result += visit_expr(vmc, binop->right);

	return result;
}

std::string VisitorCksum::visit_factor(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IFactor> factor) {
	std::string result("factor");
	if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::String>>(factor)) {
		result += std::to_string(p->is_negated());
		result += template_parser.resolve(p->factor->text(), reg);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Comparison>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_comparison(vmc, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Check>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_check(vmc, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::IExpr>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_expr(vmc, p->factor);
	} else {
		throw std::runtime_error("Unknown factor type");
	}

	return result;
}

std::string VisitorCksum::visit_comparison(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Comparison> comparison) {
	std::string result("comparison");
	result += template_parser.resolve(comparison->left->text(), reg);
	result += template_parser.resolve(comparison->right->text(), reg);
	result += comparison->op().value();
	return result;
}

std::string VisitorCksum::visit_check(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Check> check) {
	std::string result = "check";
	result += template_parser.resolve(std::string(*check->select_expr), reg);

	if (check->timeout) {
		result += template_parser.resolve(check->timeout->text(), reg);
	} else {
		auto check_timeout_found = reg->params.find("TESTO_CHECK_DEFAULT_TIMEOUT");
		result += (check_timeout_found != reg->params.end()) ? check_timeout_found->second : "1ms";
	}

	if (check->interval) {
		result += template_parser.resolve(check->interval->text(), reg);
	} else {
		auto check_interval_found = reg->params.find("TESTO_CHECK_DEFAULT_INTERVAL");
		result += (check_interval_found != reg->params.end()) ? check_interval_found->second : "1s";
	}

	return result;
}

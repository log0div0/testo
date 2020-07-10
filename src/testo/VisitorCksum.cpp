
#include "VisitorCksum.hpp"
#include "backends/Environment.hpp"
#include "IR/Program.hpp"
#include <algorithm>

uint64_t VisitorCksum::visit(std::shared_ptr<IR::Test> test) {
	std::string result = test->name();

	for (auto parent: test->parents) {
		result += parent->name();
	}

	result += test->snapshots_needed();

	StackPusher<VisitorCksum> pusher(this, test->stack);
	for (auto cmd: test->ast_node->cmd_block->commands) {
		result += visit_cmd(cmd);
	}

	std::hash<std::string> h;
	return h(result);
}

std::string VisitorCksum::visit_cmd(std::shared_ptr<AST::Cmd> cmd) {
	std::string result;

	for (auto vm_token: cmd->vms) {
		result += vm_token.value();
		result += visit_action(cmd->action);
	}
	return result;
}

std::string VisitorCksum::visit_action_block(std::shared_ptr<AST::ActionBlock> action_block) {
	std::string result("BLOCK");
	for (auto action: action_block->actions) {
		result += visit_action(action);
	}
	return result;
}

std::string VisitorCksum::visit_action(std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Abort>>(action)) {
		return visit_abort(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Print>>(action)) {
		return visit_print(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Type>>(action)) {
		return visit_type({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Wait>>(action)) {
		return visit_wait({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		return std::string(*(p->action));
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Press>>(action)) {
		return visit_press({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Hold>>(action)) {
		return std::string(*p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Release>>(action)) {
		return std::string(*p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Mouse>>(action)) {
		return visit_mouse(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Plug>>(action)) {
		return visit_plug(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Shutdown>>(action)) {
		return visit_shutdown(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Start>>(action)) {
		return "start";
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Stop>>(action)) {
		return "stop";
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Exec>>(action)) {
		return visit_exec({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Copy>>(action)) {
		return visit_copy({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroCall>>(action)) {
		return visit_macro_call(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		return visit_if_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		return visit_for_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::CycleControl>>(action)) {
		return p->action->t.value();
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		return visit_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Empty>>(action)) {
		return "";
	} else {
		throw std::runtime_error("Unknown action");
	}
}

std::string VisitorCksum::visit_abort(std::shared_ptr<AST::Abort> abort) {
	std::string result("abort");
	result += template_parser.resolve(abort->message->text(), stack);
	return result;
}

std::string VisitorCksum::visit_print(std::shared_ptr<AST::Print> print) {
	std::string result("print");
	result += template_parser.resolve(print->message->text(), stack);
	return result;
}

std::string VisitorCksum::visit_press(const IR::Press& press) {
	std::string result = std::string(*press.ast_node);
	result += press.interval();
	return result;
}

std::string VisitorCksum::visit_type(const IR::Type& type) {
	std::string result("type");
	result += type.text();;
	result += type.interval();
	return result;
}

std::string VisitorCksum::visit_wait(const IR::Wait& wait) {
	std::string result = "wait";
	result += template_parser.resolve(std::string(*wait.ast_node->select_expr), wait.stack);
	result += wait.timeout();
	result += wait.interval();
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


std::string VisitorCksum::visit_mouse_selectable(const IR::MouseSelectable& mouse_selectable) {
	std::string result = template_parser.resolve(mouse_selectable.ast_node->text(), mouse_selectable.stack);

	for (auto specifier: mouse_selectable.ast_node->specifiers) {
		result += std::string(*specifier);
	}

	result += mouse_selectable.timeout();

	return result;
}

std::string VisitorCksum::visit_mouse_move_target(std::shared_ptr<AST::IMouseMoveTarget> target) {
	std::string result;
	if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseCoordinates>>(target)) {
		result = std::string(*p->target);
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseSelectable>>(target)) {
		result = visit_mouse_selectable({p->target, stack});
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

std::string VisitorCksum::visit_plug(std::shared_ptr<AST::Plug> plug) {
	std::string result("plug");
	result += std::to_string(plug->is_on());
	result += plug->type.value();
	result += plug->name_token.value();
	if (plug->path) { //only for dvd
		fs::path path = template_parser.resolve(plug->path->text(), stack);
		if (path.is_relative()) {
			path = plug->t.begin().file.parent_path() / path;
		}
		//add signature for dvd file
		result += file_signature(path, env->content_cksum_maxsize());
	}

	return result;
}

std::string VisitorCksum::visit_shutdown(std::shared_ptr<AST::Shutdown> shutdown) {
	std::string result("shutdown");
	if (shutdown->time_interval) {
		result += shutdown->time_interval.value();
	} else {
		result += "1m";
	}
	return result;
}

std::string VisitorCksum::visit_exec(const IR::Exec& exec) {
	std::string result("exec");

	result += exec.ast_node->process_token.value();
	result += exec.text();
	result += exec.timeout();

	return result;
}

std::string VisitorCksum::visit_copy(const IR::Copy& copy) {
	std::string result(copy.ast_node->t.value());

	std::string from = copy.from();

	result += from;

	if (copy.ast_node->is_to_guest()) {
		if (!fs::exists(from)) {
			throw std::runtime_error("Specified path doesn't exist: " + from);
		}

		//now we should take care of last modify date of every file and folder in the folder
		if (fs::is_regular_file(from)) {
			result += file_signature(from, env->content_cksum_maxsize());
		} else if (fs::is_directory(from)) {
			result += directory_signature(from, env->content_cksum_maxsize());
		} else {
			throw std::runtime_error("Unknown type of file: " + from);
		}
	} //TODO: I wonder what it must be for the from copy

	result += copy.to();
	result += copy.timeout();

	return result;
}

std::string VisitorCksum::visit_macro_call(std::shared_ptr<AST::MacroCall> macro_call) {
	auto macro = IR::program->get_macro_or_throw(macro_call->name().value());

	std::map<std::string, std::string> args;

	for (size_t i = 0; i < macro_call->args.size(); ++i) {
		auto value = template_parser.resolve(macro_call->args[i]->text(), stack);
		args[macro->ast_node->args[i]->name()] = value;
	}

	for (size_t i = macro_call->args.size(); i < macro->ast_node->args.size(); ++i) {
		auto value = template_parser.resolve(macro->ast_node->args[i]->default_value->text(), stack);
		args[macro->ast_node->args[i]->name()] = value;
	}

	StackPusher<VisitorCksum> pusher(this, macro->new_stack(args));
	return visit_action_block(macro->ast_node->action_block->action);
}

std::string VisitorCksum::visit_if_clause(std::shared_ptr<AST::IfClause> if_clause) {
	std::string result("if");
	result += visit_expr(if_clause->expr);

	result += visit_action(if_clause->if_action);

	if (if_clause->has_else()) {
		result += "else";
		result += visit_action(if_clause->else_action);
	}

	return result;
}

std::string VisitorCksum::visit_range(std::shared_ptr<AST::Range> range) {
	std::string result = "RANGE";

	result += template_parser.resolve(range->r1->text(), stack);

	if (range->r2) {
		result += " ";
		result += template_parser.resolve(range->r2->text(), stack);
	}

	return result;
}

std::string VisitorCksum::visit_for_clause(std::shared_ptr<AST::ForClause> for_clause) {
	std::string result("for");
	//we should drop the counter from cksum

	if (auto p = std::dynamic_pointer_cast<AST::CounterList<AST::Range>>(for_clause->counter_list)) {
		result += visit_range(p->counter_list);
	} else {
		throw std::runtime_error("Unknown counter list");
	}

	result += visit_action(for_clause->cycle_body);
	if (for_clause->else_token) {
		result += "else";
		result += visit_action(for_clause->else_action);
	}
	return result;
}

std::string VisitorCksum::visit_expr(std::shared_ptr<AST::IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::BinOp>>(expr)) {
		return visit_binop(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::IFactor>>(expr)) {
		return visit_factor(p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

std::string VisitorCksum::visit_binop(std::shared_ptr<AST::BinOp> binop) {
	std::string result("binop");
	result += visit_expr(binop->left);
	result += visit_expr(binop->right);

	return result;
}

std::string VisitorCksum::visit_factor(std::shared_ptr<AST::IFactor> factor) {
	std::string result("factor");
	if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::String>>(factor)) {
		result += std::to_string(p->is_negated());
		result += template_parser.resolve(p->factor->text(), stack);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Comparison>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_comparison(p->factor);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Check>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_check({p->factor, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::IExpr>>(factor)) {
		result += std::to_string(p->is_negated());
		result += visit_expr(p->factor);
	} else {
		throw std::runtime_error("Unknown factor type");
	}

	return result;
}

std::string VisitorCksum::visit_comparison(std::shared_ptr<AST::Comparison> comparison) {
	std::string result("comparison");
	result += template_parser.resolve(comparison->left->text(), stack);
	result += template_parser.resolve(comparison->right->text(), stack);
	result += comparison->op().value();
	return result;
}

std::string VisitorCksum::visit_check(const IR::Check& check) {
	std::string result = "check";
	result += template_parser.resolve(std::string(*check.ast_node->select_expr), check.stack);
	result += check.timeout();
	result += check.interval();
	return result;
}

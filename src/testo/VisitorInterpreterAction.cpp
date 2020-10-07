
#include "VisitorInterpreterAction.hpp"
#include "Exceptions.hpp"
#include "IR/Program.hpp"

#include "coro/Timer.h"

static void sleep(const std::string& interval) {
	coro::Timer timer;
	timer.waitFor(std::chrono::milliseconds(time_to_milliseconds(interval)));
}


void VisitorInterpreterAction::visit_action_block(std::shared_ptr<AST::ActionBlock> action_block) {
	for (auto action: action_block->actions) {
		visit_action(action);
	}
}

void VisitorInterpreterAction::visit_abort(const IR::Abort& abort) {
	throw AbortException(abort.ast_node, current_controller, abort.message());
}

void VisitorInterpreterAction::visit_print(const IR::Print& print) {
	try {
		reporter.print(current_controller, print.message());
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(print.ast_node, current_controller));
	}
}

void VisitorInterpreterAction::visit_sleep(const IR::Sleep& sleep) {
	reporter.sleep(current_controller, sleep.timeout());
	::sleep(sleep.timeout());
}

void VisitorInterpreterAction::visit_macro_call(std::shared_ptr<AST::MacroCall> macro_call) {
	std::vector<std::pair<std::string, std::string>> args;
	std::map<std::string, std::string> vars;
	auto macro = IR::program->get_macro_or_throw(macro_call->name().value());

	for (size_t i = 0; i < macro_call->args.size(); ++i) {
		auto value = template_parser.resolve(macro_call->args[i]->text(), stack);
		vars[macro->ast_node->args[i]->name()] = value;
		args.push_back(std::make_pair(macro->ast_node->args[i]->name(), value));
	}

	for (size_t i = macro_call->args.size(); i < macro->ast_node->args.size(); ++i) {
		auto value = template_parser.resolve(macro->ast_node->args[i]->default_value->text(), stack);
		vars[macro->ast_node->args[i]->name()] = value;
		args.push_back(std::make_pair(macro->ast_node->args[i]->name(), value));
	}

	reporter.macro_call(current_controller, macro_call->name(), args);

	StackPusher<VisitorInterpreterAction> new_ctx(this, macro->new_stack(vars));
	try {
		if (auto p = std::dynamic_pointer_cast<AST::MacroBody<AST::MacroBodyAction>>(macro->ast_node->body)) {
			visit_action_block(p->macro_body->action_block->action);
		} else if (auto p = std::dynamic_pointer_cast<AST::MacroBody<AST::MacroBodyEmpty>>(macro->ast_node->body)) {
			;
		} else {
			throw std::runtime_error("Unknown macro body type");
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(MacroException(macro_call));
	}
}

void VisitorInterpreterAction::visit_if_clause(std::shared_ptr<AST::IfClause> if_clause) {
	bool expr_result;
	try {
		expr_result = visit_expr(if_clause->expr);
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(if_clause, current_controller));
	}
	//everything else should be caught at test level
	if (expr_result) {
		return visit_action(if_clause->if_action);
	} else if (if_clause->has_else()) {
		return visit_action(if_clause->else_action);
	}
}

void VisitorInterpreterAction::visit_for_clause(std::shared_ptr<AST::ForClause> for_clause) {
	uint32_t i = 0;

	std::vector<std::string> values;

	if (auto p = std::dynamic_pointer_cast<AST::CounterList<AST::Range>>(for_clause->counter_list)) {
		values = visit_range({p->counter_list, stack});
	} else {
		throw std::runtime_error("Unknown counter list type");
	}

	std::map<std::string, std::string> vars;
	for (i = 0; i < values.size(); ++i) {
		vars[for_clause->counter.value()] = values[i];

		try {
			auto new_stack = std::make_shared<StackNode>();
			new_stack->parent = stack;
			new_stack->vars = vars;
			StackPusher<VisitorInterpreterAction> new_ctx(this, new_stack);
				visit_action(for_clause->cycle_body);

		} catch (const CycleControlException& cycle_control) {
			if (cycle_control.token.type() == Token::category::break_) {
				break;
			} else if (cycle_control.token.type() == Token::category::continue_) {
				continue;
			} else {
				throw std::runtime_error(std::string("Unknown cycle control command: ") + cycle_control.token.value());
			}
		}
	}

	if ((i == values.size()) && for_clause->else_token) {
		visit_action(for_clause->else_action);
	}
}

std::vector<std::string> VisitorInterpreterAction::visit_range(const IR::Range& range) {
	return range.values();
}


bool VisitorInterpreterAction::visit_expr(std::shared_ptr<AST::IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::BinOp>>(expr)) {
		return visit_binop(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::IFactor>>(expr)) {
		return visit_factor(p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

bool VisitorInterpreterAction::visit_binop(std::shared_ptr<AST::BinOp> binop) {
	auto left = visit_expr(binop->left);

	if (binop->op().value() == "AND") {
		if (!left) {
			return left;
		} else {
			return visit_expr(binop->right);
		}
	} else if (binop->op().value() == "OR") {
		if (left) {
			return left;
		} else {
			return visit_expr(binop->right);
		}
	} else {
		throw std::runtime_error("Unknown binop operation");
	}
}

bool VisitorInterpreterAction::visit_factor(std::shared_ptr<AST::IFactor> factor) {
	bool is_negated = factor->is_negated();

	if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::String>>(factor)) {
		return is_negated ^ (bool)template_parser.resolve(p->factor->text(), stack).length();
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Comparison>>(factor)) {
		return is_negated ^ visit_comparison({p->factor, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Defined>>(factor)) {
		return is_negated ^ visit_defined({p->factor, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Check>>(factor)) {
		return is_negated ^ visit_check({p->factor, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::ParentedExpr>>(factor)) {
		return is_negated ^ visit_expr(p->factor->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::IExpr>>(factor)) {
		return is_negated ^ visit_expr(p->factor);
	} else {
		throw std::runtime_error("Unknown factor type");
	}
}

bool VisitorInterpreterAction::visit_comparison(const IR::Comparison& comparison) {
	return comparison.calculate();
}

bool VisitorInterpreterAction::visit_defined(const IR::Defined& defined) {
	return defined.is_defined();
}

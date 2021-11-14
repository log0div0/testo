
#include <coro/Timer.h>
#include "VisitorInterpreterAction.hpp"
#include "Exceptions.hpp"
#include "IR/Program.hpp"
#include "Parser.hpp"

void VisitorInterpreterAction::visit_action_block(std::shared_ptr<AST::Block<AST::Action>> action_block) {
	for (auto action: action_block->items) {
		visit_action(action);
	}
}

void VisitorInterpreterAction::visit_print(const IR::Print& print) {
	try {
		reporter.print(current_controller, print);
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(print.ast_node, current_controller));
	}
}

void VisitorInterpreterAction::visit_abort(const IR::Abort& abort) {
	reporter.abort(current_controller, abort);
	throw AbortException(abort.ast_node, current_controller, abort.message());
}

void VisitorInterpreterAction::visit_sleep(const IR::Sleep& sleep) {
	reporter.sleep(current_controller, sleep);
	coro::Timer timer;
	timer.waitFor(sleep.timeout().value());
}

void VisitorInterpreterAction::visit_macro_call(const IR::MacroCall& macro_call) {
	reporter.macro_action_call(current_controller, macro_call);
	macro_call.visit_interpreter<AST::Action>(this);
}

void VisitorInterpreterAction::visit_macro_body(const std::shared_ptr<AST::Block<AST::Action>>& macro_body) {
	visit_action_block(macro_body);
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

	if (auto p = std::dynamic_pointer_cast<AST::Range>(for_clause->counter_list)) {
		values = IR::Range({p, stack}).values();
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
				throw std::runtime_error("Unknown cycle control command: " + cycle_control.token.value());
			}
		}
	}

	if ((i == values.size()) && for_clause->else_token) {
		visit_action(for_clause->else_action);
	}
}

bool VisitorInterpreterAction::visit_expr(std::shared_ptr<AST::Expr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::BinOp>(expr)) {
		return visit_binop(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::StringExpr>(expr)) {
		return (bool)template_parser.resolve(p->str->text(), stack).length();
	} else if (auto p = std::dynamic_pointer_cast<AST::Negation>(expr)) {
		return !visit_expr(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Comparison>(expr)) {
		return visit_comparison({ p, stack });
	} else if (auto p = std::dynamic_pointer_cast<AST::Defined>(expr)) {
		return visit_defined({ p, stack });
	} else if (auto p = std::dynamic_pointer_cast<AST::Check>(expr)) {
		return visit_check({ p, stack });
	} else if (auto p = std::dynamic_pointer_cast<AST::ParentedExpr>(expr)) {
		return visit_expr(p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

bool VisitorInterpreterAction::visit_binop(std::shared_ptr<AST::BinOp> binop) {
	auto left = visit_expr(binop->left);

	if (binop->op.value() == "AND") {
		if (!left) {
			return left;
		} else {
			return visit_expr(binop->right);
		}
	} else if (binop->op.value() == "OR") {
		if (left) {
			return left;
		} else {
			return visit_expr(binop->right);
		}
	} else {
		throw std::runtime_error("Unknown binop operation");
	}
}

bool VisitorInterpreterAction::visit_comparison(const IR::Comparison& comparison) {
	return comparison.calculate();
}

bool VisitorInterpreterAction::visit_defined(const IR::Defined& defined) {
	return defined.is_defined();
}

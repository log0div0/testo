
#pragma once

#include "IR/Action.hpp"
#include "IR/Expr.hpp"
#include "IR/Controller.hpp"
#include "IR/Macro.hpp"
#include "IR/Expr.hpp"
#include "Reporter.hpp"
#include "TemplateLiterals.hpp"

struct ActionException: ExceptionWithPos {
	ActionException(const std::shared_ptr<AST::Node>& node, const std::shared_ptr<IR::Controller>& controller):
		ExceptionWithPos(node->begin(), {})
	{
		msg = "Error while performing action " + node->to_string();
		if (controller) {
			msg += " on " + controller->type() + " " + controller->name();
		}
	}
};

struct AbortException: ExceptionWithPos {
	AbortException(const std::shared_ptr<AST::Abort>& node, const std::shared_ptr<IR::Controller>& controller, const std::string& message):
		ExceptionWithPos(node->begin(), {})
	{
		msg = "Caught abort action ";
		if (controller) {
			msg += "on " + controller->type() + " " +  controller->name();
		}
		msg += " with message: ";
		msg += message;
	}
};

struct CycleControlException: std::exception {
	CycleControlException(Token token_): token(std::move(token_))
	{
	}

	Token token;
};

struct VisitorInterpreterAction {
	VisitorInterpreterAction(std::shared_ptr<IR::Controller> controller, std::shared_ptr<StackNode> stack, Reporter& reporter):
		current_controller(controller), stack(stack), reporter(reporter) {}

	virtual ~VisitorInterpreterAction() {}

	virtual void visit_action(std::shared_ptr<AST::Action> action) = 0;
	virtual void visit_copy(const IR::Copy& copy) = 0;
	virtual bool visit_check(const IR::Check& check) = 0;

	void visit_action_block(std::shared_ptr<AST::Block<AST::Action>> action_block);
	void visit_print(const IR::Print& print);
	void visit_abort(const IR::Abort& abort);
	void visit_sleep(const IR::Sleep& sleep);
	void visit_macro_call(const IR::MacroCall& macro_call);
	void visit_macro_body(const std::shared_ptr<AST::Block<AST::Action>>& macro_body);
	void visit_if_clause(std::shared_ptr<AST::IfClause> if_clause);
	void visit_for_clause(std::shared_ptr<AST::ForClause> for_clause);

	bool visit_expr(std::shared_ptr<AST::Expr> expr);
	bool visit_binop(std::shared_ptr<AST::BinOp> binop);
	bool visit_string_expr(const IR::StringExpr& string_expr);
	bool visit_comparison(const IR::Comparison& comparison);
	bool visit_defined(const IR::Defined& defined);

	std::shared_ptr<IR::Controller> current_controller;
	std::shared_ptr<StackNode> stack;
	Reporter& reporter;
};
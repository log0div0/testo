
#pragma once

#include "IR/Action.hpp"
#include "IR/Expr.hpp"
#include "IR/Controller.hpp"
#include "IR/Macro.hpp"
#include "IR/Expr.hpp"
#include "Reporter.hpp"
#include "TemplateLiterals.hpp"

struct VisitorInterpreterAction {
	VisitorInterpreterAction(std::shared_ptr<IR::Controller> controller, std::shared_ptr<StackNode> stack, Reporter& reporter):
		current_controller(controller), stack(stack), reporter(reporter) {}

	virtual ~VisitorInterpreterAction() {}

	virtual void visit_action(std::shared_ptr<AST::Action> action) = 0;
	virtual void visit_copy(const IR::Copy& copy) = 0;
	virtual bool visit_check(const IR::Check& check) = 0;
	virtual void visit_abort(const IR::Abort& abort) = 0;

	void visit_action_block(std::shared_ptr<AST::ActionBlock> action_block);
	void visit_print(const IR::Print& print);
	void visit_sleep(const IR::Sleep& sleep);
	void visit_macro_call(const IR::MacroCall& macro_call);
	void visit_macro_body(const std::shared_ptr<AST::MacroBodyAction>& macro_body);
	void visit_if_clause(std::shared_ptr<AST::IfClause> if_clause);
	void visit_for_clause(std::shared_ptr<AST::ForClause> for_clause);
	std::vector<std::string> visit_range(const IR::Range& range);

	bool visit_expr(std::shared_ptr<AST::Expr> expr);
	bool visit_binop(std::shared_ptr<AST::BinOp> binop);
	bool visit_comparison(const IR::Comparison& comparison);
	bool visit_defined(const IR::Defined& defined);

	template_literals::Parser template_parser;
	std::shared_ptr<IR::Controller> current_controller;
	std::shared_ptr<StackNode> stack;
	Reporter& reporter;
};

#pragma once

#include "Node.hpp"
#include "Register.hpp"
#include "VmController.hpp"

struct VisitorCksum {
	VisitorCksum(Register& reg): reg(reg) {}

	uint64_t visit(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Snapshot> action_block);
	std::string visit_action_block(std::shared_ptr<VmController> vm, std::shared_ptr<AST::ActionBlock> action_block);
	std::string visit_action(std::shared_ptr<VmController> vm, std::shared_ptr<AST::IAction> action);
	std::string visit_type(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Type> type);
	std::string visit_wait(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Wait> wait);
	std::string visit_press(std::shared_ptr<AST::Press> press);
	std::string visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec);
	std::string visit_plug(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	std::string visit_exec(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Exec> exec);
	std::string visit_set(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Set> set);
	std::string visit_copyto(std::shared_ptr<VmController> vm, std::shared_ptr<AST::CopyTo> copyto);
	std::string visit_macro_call(std::shared_ptr<VmController> vm, std::shared_ptr<AST::MacroCall> macro_call);
	std::string visit_if_clause(std::shared_ptr<VmController> vm, std::shared_ptr<AST::IfClause> if_clause);

	std::string visit_expr(std::shared_ptr<VmController> vm, std::shared_ptr<AST::IExpr> expr);
	std::string visit_binop(std::shared_ptr<VmController> vm, std::shared_ptr<AST::BinOp> binop);
	std::string visit_factor(std::shared_ptr<VmController> vm, std::shared_ptr<AST::IFactor> factor);
	std::string resolve_var(std::shared_ptr<VmController> vm, const std::string& var);
	std::string visit_word(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Word> word);
	std::string visit_comparison(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Comparison> comparison);

	Register& reg;
};

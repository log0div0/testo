
#pragma once

#include "Node.hpp"
#include "Register.hpp"
#include "backends/Environment.hpp"

#include <set>

struct VisitorSemantic {
	VisitorSemantic(Register& reg, Environment& env);

	void visit(std::shared_ptr<AST::Program> program);
	void visit_stmt(std::shared_ptr<AST::IStmt> stmt);
	void visit_macro(std::shared_ptr<AST::Macro> macro);
	void visit_test(std::shared_ptr<AST::Test> test);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::Cmd> cmd);
	void visit_action_block(std::shared_ptr<AST::ActionBlock> action_block);
	void visit_action(std::shared_ptr<AST::IAction> action);
	void visit_press(std::shared_ptr<AST::Press> press);
	void visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec);
	void visit_plug(std::shared_ptr<AST::Plug> plug);
	void visit_exec(std::shared_ptr<AST::Exec> exec);
	void visit_macro_call(std::shared_ptr<AST::MacroCall> macro_call);
	void visit_for_clause(std::shared_ptr<AST::ForClause> for_clause);

	void visit_controller(std::shared_ptr<AST::Controller> controller);
	void visit_machine(std::shared_ptr<AST::Controller> machine);
	void visit_flash(std::shared_ptr<AST::Controller> flash); //flash drive
	nlohmann::json visit_attr_block(std::shared_ptr<AST::AttrBlock> attr_block, const std::string& ctx);
	std::string resolve_var(const std::string& var);
	std::string visit_word(std::shared_ptr<AST::Word> word);
	void visit_attr(std::shared_ptr<AST::Attr> attr, nlohmann::json& config, const std::string& ctx);

	bool is_button(const Token& t) const;
	std::set<std::string> keys;

	//bool is for "requires a name"
	using attr_ctx = std::unordered_map<std::string, std::pair<bool, Token::category>>;
	std::unordered_map<std::string, attr_ctx> attr_ctxs;

	Register& reg;
	Environment& env;
};

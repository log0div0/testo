
#pragma once

#include <Lexer.hpp>
#include <Node.hpp>
#include <Utils.hpp>
#include <set>

struct Parser {
	Parser(const fs::path& file);

	std::shared_ptr<AST::Program> parse();
private:

	struct Ctx {
		Ctx(const fs::path& file): lex(file) {}
		Lexer lex;
		std::array<Token, 2> lookahead;
		size_t p = 0; //current position in lookahead buffer
	};

	//inner helpers
	void match(Token::category type);
	void match(std::vector<Token::category> types);
	void consume();

	Token LT(size_t i) const;
	Token::category LA(size_t i) const;

	bool test_assignment() const;
	bool test_stmt() const;
	bool test_controller() const;
	bool test_command() const;
	bool test_action() const;
	bool test_include() const;
	bool test_term() const;
	bool test_comparison() const;
	bool is_button(const Token& t) const;

	void newline_list();

	void handle_include();
	std::shared_ptr<AST::IStmt> stmt();
	std::shared_ptr<AST::Stmt<AST::Snapshot>> snapshot();
	std::shared_ptr<AST::Stmt<AST::Test>> test();
	std::shared_ptr<AST::Stmt<AST::Macro>> macro();
	std::shared_ptr<AST::VmState> vm_state();
	std::shared_ptr<AST::Assignment> assignment();
	std::shared_ptr<AST::Attr> attr();
	std::shared_ptr<AST::AttrBlock> attr_block();
	std::shared_ptr<AST::Stmt<AST::Controller>> controller();
	std::shared_ptr<AST::Cmd> command();
	std::shared_ptr<AST::CmdBlock> command_block();
	std::shared_ptr<AST::KeySpec> key_spec();
	std::shared_ptr<AST::IAction> action();
	std::shared_ptr<AST::Action<AST::Type>> type();
	std::shared_ptr<AST::Action<AST::Wait>> wait();
	std::shared_ptr<AST::Action<AST::Press>> press();
	std::shared_ptr<AST::Action<AST::Plug>> plug();
	std::shared_ptr<AST::Action<AST::Start>> start();
	std::shared_ptr<AST::Action<AST::Stop>> stop();
	std::shared_ptr<AST::Action<AST::Exec>> exec();
	std::shared_ptr<AST::Action<AST::Set>> set();
	std::shared_ptr<AST::Action<AST::CopyTo>> copyto();
	std::shared_ptr<AST::Action<AST::ActionBlock>> action_block();
	std::shared_ptr<AST::Action<AST::MacroCall>> macro_call();
	std::shared_ptr<AST::Action<AST::IfClause>> if_clause();

	//expressions
	std::shared_ptr<AST::Term> term();
	std::shared_ptr<AST::IFactor> factor();
	std::shared_ptr<AST::Comparison> comparison();
	std::shared_ptr<AST::Expr<AST::BinOp>> binop(std::shared_ptr<AST::IExpr> left);
	std::shared_ptr<AST::IExpr> expr();

	std::vector<Ctx> lexers;
};

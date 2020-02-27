
#pragma once

#include "Lexer.hpp"
#include "Node.hpp"
#include "Utils.hpp"
#include "Token.hpp"
#include <set>
#include <array>

struct Parser {
	Parser() = default;
	Parser(const fs::path& file, const std::string& input);

	std::shared_ptr<AST::Program> parse();
private:

	struct Ctx {
		Ctx(const fs::path& file, const std::string& input): lex(file, input) {}
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

	bool test_stmt() const;
	bool test_controller() const;
	bool test_test() const;
	bool test_command() const;
	bool test_action() const;
	bool test_include() const;
	bool test_string() const;
	bool test_selectable() const;
	bool test_select_expr() const;
	bool test_binary() const;
	bool test_comparison() const;
	bool is_button(const Token& t) const;

	void newline_list();

	void handle_include();
	std::shared_ptr<AST::IStmt> stmt();
	std::shared_ptr<AST::Stmt<AST::Test>> test();
	std::shared_ptr<AST::Stmt<AST::Macro>> macro();
	std::shared_ptr<AST::Stmt<AST::Param>> param();
	std::shared_ptr<AST::Attr> attr();
	std::shared_ptr<AST::AttrBlock> attr_block();
	std::shared_ptr<AST::Stmt<AST::Controller>> controller();
	std::shared_ptr<AST::Cmd> command();
	std::shared_ptr<AST::CmdBlock> command_block();
	std::shared_ptr<AST::KeySpec> key_spec();
	std::shared_ptr<AST::IAction> action();
	std::shared_ptr<AST::Action<AST::Empty>> empty_action();
	std::shared_ptr<AST::Action<AST::Abort>> abort();
	std::shared_ptr<AST::Action<AST::Print>> print();
	std::shared_ptr<AST::Action<AST::Type>> type();
	std::shared_ptr<AST::Action<AST::Wait>> wait();
	std::shared_ptr<AST::Action<AST::Sleep>> sleep();
	std::shared_ptr<AST::Action<AST::Press>> press();
	std::shared_ptr<AST::Action<AST::Mouse>> mouse();
	std::shared_ptr<AST::MouseEvent<AST::MouseMoveClick>> mouse_move_click();
	std::shared_ptr<AST::MouseEvent<AST::MouseHold>> mouse_hold();
	std::shared_ptr<AST::MouseEvent<AST::MouseRelease>> mouse_release();
	std::shared_ptr<AST::MouseEvent<AST::MouseWheel>> mouse_wheel();
	std::shared_ptr<AST::MouseMoveTarget<AST::MouseCoordinates>> mouse_coordinates();
	std::shared_ptr<AST::Action<AST::Plug>> plug();
	std::shared_ptr<AST::Action<AST::Start>> start();
	std::shared_ptr<AST::Action<AST::Stop>> stop();
	std::shared_ptr<AST::Action<AST::Shutdown>> shutdown();
	std::shared_ptr<AST::Action<AST::Exec>> exec();
	std::shared_ptr<AST::Action<AST::Copy>> copy();
	std::shared_ptr<AST::Action<AST::ActionBlock>> action_block();
	std::shared_ptr<AST::Action<AST::MacroCall>> macro_call();
	std::shared_ptr<AST::Action<AST::IfClause>> if_clause();
	std::shared_ptr<AST::Action<AST::ForClause>> for_clause();
	std::shared_ptr<AST::Action<AST::CycleControl>> cycle_control();

	//expressions
	std::shared_ptr<AST::ISelectExpr> select_expr();
	std::shared_ptr<AST::SelectExpr<AST::SelectUnOp>> select_unop();
	std::shared_ptr<AST::SelectExpr<AST::SelectParentedExpr>> select_parented_expr();
	std::shared_ptr<AST::SelectExpr<AST::SelectBinOp>> select_binop(std::shared_ptr<AST::ISelectExpr> left);

	std::shared_ptr<AST::ISelectable> selectable();
	std::shared_ptr<AST::Selectable<AST::SelectJS>> select_js();

	std::shared_ptr<AST::String> string();

	std::shared_ptr<AST::IFactor> factor();
	std::shared_ptr<AST::Check> check();
	std::shared_ptr<AST::Comparison> comparison();
	std::shared_ptr<AST::Expr<AST::BinOp>> binop(std::shared_ptr<AST::IExpr> left);
	std::shared_ptr<AST::IExpr> expr();

	std::vector<Ctx> lexers;

	std::vector<fs::path> already_included;
};

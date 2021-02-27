
#pragma once

#include "Lexer.hpp"
#include "AST.hpp"
#include "Utils.hpp"
#include "Token.hpp"
#include <set>
#include <array>

constexpr static size_t LOOKAHEAD_BUFFER_SIZE = 2;

struct Parser {
	static Parser load_dir(const fs::path& dir);
	static Parser load_file(const fs::path& file);
	static Parser load(const fs::path& path);

	Parser() = default;
	Parser(const fs::path& file, const std::string& input);
	Parser(const std::vector<Token>& tokens);

	std::shared_ptr<AST::Program> parse();
	std::shared_ptr<AST::CmdBlock> command_block();
	std::shared_ptr<AST::Action<AST::ActionBlock>> action_block();
	std::shared_ptr<AST::StmtBlock> stmt_block();
private:

	struct Ctx {
		Ctx(const fs::path& file, const std::string& input) {
			Lexer lex(file, input);
			Token t;
			for (t = lex.get_next_token(); t.type() != Token::category::eof; t = lex.get_next_token()) {
				tokens.push_back(t);
			}

			tokens.push_back(t);
		}

		Ctx(const std::vector<Token>& tokens): tokens(tokens) {}
		std::vector<Token> tokens;
		size_t p = 0; //current position in tokens buffer
	};

	//inner helpers
	void match(Token::category type);
	void match(std::vector<Token::category> types);
	void consume();

	Token LT(size_t i) const;
	Token::category LA(size_t i) const;

	bool test_stmt() const;
	bool test_macro_call() const;
	bool test_controller() const;
	bool test_test() const;
	bool test_command(size_t index = 1) const;
	bool test_action(size_t index = 1) const;
	bool test_counter_list() const;
	bool test_include() const;
	bool test_string(size_t index = 1) const;
	bool test_selectable() const;
	bool test_comparison() const;
	bool test_defined() const;
	bool is_button(const Token& t) const;

	void newline_list();

	void handle_include();
	std::shared_ptr<AST::IStmt> stmt();
	std::shared_ptr<AST::Stmt<AST::Test>> test();
	std::shared_ptr<AST::MacroArg> macro_arg();
	std::vector<Token> macro_body(const std::string& name);
	std::shared_ptr<AST::Stmt<AST::Macro>> macro();
	std::shared_ptr<AST::Stmt<AST::Param>> param();
	std::shared_ptr<AST::Attr> attr(const std::string& ctx_name);
	std::shared_ptr<AST::AttrBlock> attr_block(const std::string& ctx_name);
	std::shared_ptr<AST::Stmt<AST::Controller>> controller();
	std::shared_ptr<AST::ICmd> command();
	std::shared_ptr<AST::KeyCombination> key_combination();
	std::shared_ptr<AST::KeySpec> key_spec();
	std::shared_ptr<AST::IAction> action();
	std::shared_ptr<AST::Action<AST::Empty>> empty_action();
	std::shared_ptr<AST::Action<AST::Abort>> abort();
	std::shared_ptr<AST::Action<AST::Print>> print();
	std::shared_ptr<AST::Action<AST::Type>> type();
	std::shared_ptr<AST::Action<AST::Wait>> wait();
	std::shared_ptr<AST::Action<AST::Sleep>> sleep();
	std::shared_ptr<AST::Action<AST::Press>> press();
	std::shared_ptr<AST::Action<AST::Hold>> hold();
	std::shared_ptr<AST::Action<AST::Release>> release();
	std::shared_ptr<AST::Action<AST::Mouse>> mouse();
	std::shared_ptr<AST::MouseEvent<AST::MouseMoveClick>> mouse_move_click();
	std::shared_ptr<AST::MouseEvent<AST::MouseHold>> mouse_hold();
	std::shared_ptr<AST::MouseEvent<AST::MouseRelease>> mouse_release();
	std::shared_ptr<AST::MouseEvent<AST::MouseWheel>> mouse_wheel();
	std::shared_ptr<AST::MouseAdditionalSpecifier> mouse_additional_specifier();
	std::shared_ptr<AST::MouseMoveTarget<AST::MouseSelectable>> mouse_selectable();
	std::shared_ptr<AST::MouseMoveTarget<AST::MouseCoordinates>> mouse_coordinates();
	std::shared_ptr<AST::IPlugResource> plug_resource();
	std::shared_ptr<AST::PlugResource<AST::PlugNIC>> plug_resource_nic();
	std::shared_ptr<AST::PlugResource<AST::PlugLink>> plug_resource_link();
	std::shared_ptr<AST::PlugResource<AST::PlugFlash>> plug_resource_flash();
	std::shared_ptr<AST::PlugResource<AST::PlugDVD>> plug_resource_dvd();
	std::shared_ptr<AST::PlugResource<AST::PlugHostDev>> plug_resource_hostdev();
	std::shared_ptr<AST::Action<AST::Plug>> plug();
	std::shared_ptr<AST::Action<AST::Start>> start();
	std::shared_ptr<AST::Action<AST::Stop>> stop();
	std::shared_ptr<AST::Action<AST::Shutdown>> shutdown();
	std::shared_ptr<AST::Action<AST::Exec>> exec();
	std::shared_ptr<AST::Action<AST::Copy>> copy();
	std::shared_ptr<AST::MacroCall> macro_call();
	std::shared_ptr<AST::Action<AST::IfClause>> if_clause();
	std::shared_ptr<AST::ICounterList> counter_list();
	std::shared_ptr<AST::CounterList<AST::Range>> range();
	std::shared_ptr<AST::Action<AST::ForClause>> for_clause();
	std::shared_ptr<AST::Action<AST::CycleControl>> cycle_control();

	//expressions
	std::shared_ptr<AST::ISelectExpr> select_expr();
	std::shared_ptr<AST::SelectParentedExpr> select_parented_expr();
	std::shared_ptr<AST::SelectExpr<AST::SelectBinOp>> select_binop(std::shared_ptr<AST::ISelectExpr> left);

	std::shared_ptr<AST::ISelectable> selectable();
	std::shared_ptr<AST::SelectJS> select_js();
	std::shared_ptr<AST::SelectImg> select_img();
	std::shared_ptr<AST::SelectHomm3> select_homm3();
	std::shared_ptr<AST::SelectText> select_text();

	std::shared_ptr<AST::String> string();
	std::shared_ptr<AST::StringTokenUnion> string_token_union(Token::category expected_token_type);

	std::shared_ptr<AST::IFactor> factor();
	std::shared_ptr<AST::Check> check();
	std::shared_ptr<AST::Comparison> comparison();
	std::shared_ptr<AST::Defined> defined();
	std::shared_ptr<AST::Expr<AST::BinOp>> binop(std::shared_ptr<AST::IExpr> left);
	std::shared_ptr<AST::IExpr> expr();
	std::shared_ptr<AST::ParentedExpr> parented_expr();

	std::vector<Ctx> lexers;

	std::vector<fs::path> already_included;
};

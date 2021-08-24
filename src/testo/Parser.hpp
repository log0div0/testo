
#pragma once

#include "Lexer.hpp"
#include "AST.hpp"
#include "Utils.hpp"
#include "Token.hpp"
#include <set>
#include <array>

struct Parser {
	static Parser load_dir(const fs::path& dir);
	static Parser load_file(const fs::path& file);
	static Parser load(const fs::path& path);

	Parser() = default;
	Parser(const fs::path& file, const std::string& input);
	Parser(const std::vector<Token>& tokens);

	std::shared_ptr<AST::Program> parse();
	std::shared_ptr<AST::Block<AST::Cmd>> command_block();
	std::shared_ptr<AST::Block<AST::Action>> action_block();
	std::shared_ptr<AST::Block<AST::Stmt>> stmt_block();

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
	Token eat(Token::category type);
	Token eat(std::vector<Token::category> types);

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

	using OptionName = std::string;
	using OptionValue = std::shared_ptr<AST::Node>;
	using OptionSeqSchema = std::map<OptionName, std::function<OptionValue()>>;
	std::shared_ptr<AST::OptionSeq> option_seq(const OptionSeqSchema& schema);

	void handle_include();
	std::shared_ptr<AST::Stmt> stmt();
	std::shared_ptr<AST::Test> test();
	std::shared_ptr<AST::MacroArg> macro_arg();
	std::vector<Token> macro_body(const std::string& name);
	std::shared_ptr<AST::Macro> macro();
	std::shared_ptr<AST::Param> param();

	using AttrName = std::string;
	using AttrValue = std::shared_ptr<AST::Node>;
	struct AttrDesc {
		bool id_required;
		std::function<AttrValue()> cb;
	};
	using AttrBlockSchema = std::map<AttrName, AttrDesc>;
	std::shared_ptr<AST::Attr> attr(const AttrBlockSchema& schema);
	std::shared_ptr<AST::AttrBlock> attr_block(const AttrBlockSchema& schema);

	std::shared_ptr<AST::Controller> controller();
	std::shared_ptr<AST::Cmd> command();
	std::shared_ptr<AST::IKeyCombination> key_combination();
	std::shared_ptr<AST::KeySpec> key_spec();
	std::shared_ptr<AST::Action> action();
	std::shared_ptr<AST::Empty> empty_action();
	std::shared_ptr<AST::Abort> abort();
	std::shared_ptr<AST::Print> print();
	std::shared_ptr<AST::Type> type();
	std::shared_ptr<AST::Wait> wait();
	std::shared_ptr<AST::Sleep> sleep();
	std::shared_ptr<AST::Press> press();
	std::shared_ptr<AST::Hold> hold();
	std::shared_ptr<AST::Release> release();
	std::shared_ptr<AST::Mouse> mouse();
	std::shared_ptr<AST::MouseMoveClick> mouse_move_click();
	std::shared_ptr<AST::MouseHold> mouse_hold();
	std::shared_ptr<AST::MouseRelease> mouse_release();
	std::shared_ptr<AST::MouseWheel> mouse_wheel();
	std::shared_ptr<AST::MouseAdditionalSpecifier> mouse_additional_specifier();
	std::shared_ptr<AST::MouseSelectable> mouse_selectable();
	std::shared_ptr<AST::MouseCoordinates> mouse_coordinates();
	std::shared_ptr<AST::PlugResource> plug_resource();
	std::shared_ptr<AST::PlugNIC> plug_resource_nic();
	std::shared_ptr<AST::PlugLink> plug_resource_link();
	std::shared_ptr<AST::PlugFlash> plug_resource_flash();
	std::shared_ptr<AST::PlugDVD> plug_resource_dvd();
	std::shared_ptr<AST::PlugHostDev> plug_resource_hostdev();
	std::shared_ptr<AST::Plug> plug();
	std::shared_ptr<AST::Start> start();
	std::shared_ptr<AST::Stop> stop();
	std::shared_ptr<AST::Shutdown> shutdown();
	std::shared_ptr<AST::Exec> exec();
	std::shared_ptr<AST::Copy> copy();
	std::shared_ptr<AST::Screenshot> screenshot();
	template <typename BaseType>
	std::shared_ptr<AST::MacroCall<BaseType>> macro_call();
	std::shared_ptr<AST::IfClause> if_clause();
	std::shared_ptr<AST::CounterList> counter_list();
	std::shared_ptr<AST::Range> range();
	std::shared_ptr<AST::ForClause> for_clause();
	std::shared_ptr<AST::CycleControl> cycle_control();

	//expressions
	std::shared_ptr<AST::SelectExpr> select_expr();
	std::shared_ptr<AST::BasicSelectExpr> basic_select_expr();
	std::shared_ptr<AST::SelectSimpleExpr> select_simple_expr();
	std::shared_ptr<AST::SelectParentedExpr> select_parented_expr();
	std::shared_ptr<AST::SelectBinOp> select_binop(std::shared_ptr<AST::SelectExpr> left);

	std::shared_ptr<AST::SelectJS> select_js();
	std::shared_ptr<AST::SelectImg> select_img();
	std::shared_ptr<AST::SelectHomm3> select_homm3();
	std::shared_ptr<AST::SelectText> select_text();

	std::shared_ptr<AST::String> string();
	template <Token::category category>
	std::shared_ptr<AST::ISingleToken<category>> single_token() {
		if (!test_string() && LA(1) != category) {
			throw std::runtime_error(std::string(LT(1).begin()) + ": Error: expected a string or " + Token::type_to_string(category) + ", but got " +
				Token::type_to_string(LA(1)) + " " + LT(1).value());
		}

		if (LA(1) == category) {
			return std::make_shared<AST::SingleToken<category>>(eat(category));
		}

		return std::make_shared<AST::Unparsed<AST::ISingleToken<category>>>(string());
	}

	std::shared_ptr<AST::Number> number();
	std::shared_ptr<AST::Id> id();
	std::shared_ptr<AST::TimeInterval> time_interval();
	std::shared_ptr<AST::Size> size();
	std::shared_ptr<AST::Boolean> boolean();

	std::shared_ptr<AST::Check> check();
	std::shared_ptr<AST::Comparison> comparison();
	std::shared_ptr<AST::Defined> defined();
	std::shared_ptr<AST::BinOp> binop(std::shared_ptr<AST::Expr> left);
	std::shared_ptr<AST::Expr> expr();
	std::shared_ptr<AST::SimpleExpr> simple_expr();
	std::shared_ptr<AST::ParentedExpr> parented_expr();
	std::shared_ptr<AST::Negation> negation();

	std::vector<Ctx> lexers;

	std::vector<fs::path> already_included;
};

namespace AST {

template <Token::category category>
std::shared_ptr<ISingleToken<category>> ISingleToken<category>::from_string(const std::string& str) {
	return Parser(".", str).single_token<category>();
}

inline std::shared_ptr<IKeyCombination> IKeyCombination::from_string(const std::string& str) {
	return Parser(".", str).key_combination();
}

}

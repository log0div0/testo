
#include "Parser.hpp"
#include "Utils.hpp"
#include "TemplateLiterals.hpp"
#include <fstream>
#include <fmt/format.h>

using namespace AST;

std::string generate_script(const fs::path& folder, const fs::path& current_prefix = ".") {
	std::string result("");
	for (auto& file: fs::directory_iterator(folder)) {
		if (fs::is_regular_file(file)) {
			if (fs::path(file).extension() == ".testo") {
				result += fmt::format("include \"{}\"\n", fs::path(current_prefix / fs::path(file).filename()).generic_string());
			}
		} else if (fs::is_directory(file)) {
			result += generate_script(file, current_prefix / fs::path(file).filename());
		} else {
			throw std::runtime_error("Unknown type of file: " + fs::path(file).generic_string());
		}
	}

	return result;
}

Parser Parser::load_dir(const fs::path& dir) {
	return Parser(dir, generate_script(dir));
}

Parser Parser::load_file(const fs::path& file) {
	std::ifstream input_stream(file);

	if (!input_stream) {
		throw std::runtime_error("Can't open file: " + file.generic_string());
	}

	std::string input = std::string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());

	return Parser(file, input);
}

Parser Parser::load(const fs::path& path) {
	if (fs::is_regular_file(path)) {
		return load_file(path);
	} else if (fs::is_directory(path)) {
		return load_dir(path);
	} else {
		throw std::runtime_error(std::string("Fatal error: unknown target type: ") + path.generic_string());
	}

}

Parser::Parser(const fs::path& file, const std::string& input)
{
	Ctx ctx(file, input);
	lexers.push_back(ctx);
}

Parser::Parser(const std::vector<Token>& tokens) {
	Ctx ctx(tokens);
	lexers.push_back(ctx);
}

void Parser::consume() {
	Ctx& current_lexer = lexers.back();
	current_lexer.p++;
}

void Parser::match(Token::category type) {
	if (LA(1) == type) {
		consume();
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) +
			": Error: unexpected token \"" +
			LT(1).value() + "\", expected: " + Token::type_to_string(type)); //TODO: more informative what we expected
	}
}

void Parser::match(const std::vector<Token::category> types) {
	for (auto type: types) {
		if (LA(1) == type) {
			consume();
			return;
		}
	}

	throw std::runtime_error(std::string(LT(1).begin()) +
			": Error: unexpected token \"" +
			LT(1).value() + "\""); //TODO: more informative what we expected
}

Token Parser::LT(size_t i) const {
	return lexers.back().tokens[lexers.back().p + i - 1];
}

Token::category Parser::LA(size_t i) const {
	return LT(i).type();
}

bool Parser::test_stmt() const {
	return ((LA(1) == Token::category::macro) ||
		(LA(1) == Token::category::param) ||
		test_controller() ||
		test_test() ||
		test_macro_call());
}

bool Parser::test_macro_call() const {
	return (LA(1) == Token::category::id && LA(2) == Token::category::lparen);
}

bool Parser::test_include() const {
	return (LA(1) == Token::category::include);
}

bool Parser::test_controller() const {
	return (LA(1) == Token::category::machine) ||
		(LA(1) == Token::category::flash) ||
		(LA(1) == Token::category::network);
}

bool Parser::test_test() const {
	return (LA(1) == Token::category::test) ||
		(LA(1) == Token::category::lbracket);
}

bool Parser::test_command(size_t index) const {
	return (LA(index) == Token::category::id ||
		test_string(index));
}

bool Parser::test_action(size_t index) const {
	return ((LA(index) == Token::category::abort) ||
		(LA(index) == Token::category::print) ||
		(LA(index) == Token::category::type_) ||
		(LA(index) == Token::category::wait) ||
		(LA(index) == Token::category::sleep) ||
		(LA(index) == Token::category::press) ||
		(LA(index) == Token::category::hold) ||
		(LA(index) == Token::category::release) ||
		(LA(index) == Token::category::mouse) ||
		(LA(index) == Token::category::plug) ||
		(LA(index) == Token::category::unplug) ||
		(LA(index) == Token::category::start) ||
		(LA(index) == Token::category::stop) ||
		(LA(index) == Token::category::shutdown) ||
		(LA(index) == Token::category::exec) ||
		(LA(index) == Token::category::copyto) ||
		(LA(index) == Token::category::copyfrom) ||
		(LA(index) == Token::category::screenshot) ||
		(LA(index) == Token::category::lbrace) ||
		(LA(index) == Token::category::if_) ||
		(LA(index) == Token::category::for_) ||
		(LA(index) == Token::category::break_) ||
		(LA(index) == Token::category::continue_) ||
		(LA(index) == Token::category::semi) ||
		(LA(index) == Token::category::id)); //macro call
}

bool Parser::test_counter_list() const {
	return LA(1) == Token::category::RANGE;
}

bool Parser::test_string(size_t index) const {
	return ((LA(index) == Token::category::quoted_string) ||
		(LA(index) == Token::category::triple_quoted_string));
}

bool Parser::test_selectable() const {
	return (test_string() || (LA(1) == Token::category::js)  ||
		(LA(1) == Token::category::img) ||
		(LA(1) == Token::category::homm3) ||
		(LA(1) == Token::category::exclamation_mark) ||
		(LA(1) == Token::category::lparen));
}

bool Parser::test_comparison() const {
	if (test_string()) {
		if ((LA(2) == Token::category::LESS) ||
			(LA(2) == Token::category::GREATER) ||
			(LA(2) == Token::category::EQUAL) ||
			(LA(2) == Token::category::STRLESS) ||
			(LA(2) == Token::category::STRGREATER) ||
			(LA(2) == Token::category::STREQUAL))
		{
			return true;
		}
	}
	return false;
}

bool Parser::test_defined() const {
	return LA(1) == Token::category::DEFINED;
}

void Parser::newline_list() {
	while (LA(1) == Token::category::newline) {
		match (Token::category::newline);
	}
}

void Parser::handle_include() {
	//Get new Lexer
	auto include_token = LT(1);
	match(Token::category::include);

	auto dest_file_token = LT(1);
	match(Token::category::quoted_string);
	match(Token::category::newline);
	fs::path dest_file = dest_file_token.value().substr(1, dest_file_token.value().length() - 2);

	if (dest_file.is_relative()) {
		auto current_path = lexers.back().tokens[0].begin().file;
		fs::path combined;
		if (fs::is_regular_file(current_path)) {
			combined = current_path.parent_path() / dest_file;
		} else if (fs::is_directory(current_path)) {
			combined = current_path / dest_file;
		} else {
			throw std::runtime_error("Handle include error");
		}

		if (!fs::exists(combined)) {
			throw std::runtime_error(std::string(dest_file_token.begin()) + ": fatal error: no such file: " + dest_file.generic_string());
		}
		dest_file = fs::canonical(combined);
	}

	//check for cycles

	for (auto& path: already_included) {
		if (path == dest_file) {
			return; //implementing #pramga once
		}
	}

	std::ifstream input_stream(dest_file);

	if (!input_stream) {
		throw std::runtime_error("Can't open file: " + dest_file.generic_string());
	}

	auto input = std::string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());

	Ctx new_ctx(dest_file, input);
	lexers.push_back(new_ctx);
	already_included.push_back(dest_file);
}

std::shared_ptr<Program> Parser::parse() {
	std::vector<std::shared_ptr<Stmt>> stmts;

	//we expect include command only between the declarations
	while (!lexers.empty()) {
		newline_list();
		if (LA(1) == Token::category::eof) {
			lexers.pop_back();
		} else if (test_stmt()) {
			stmts.push_back(stmt());
			newline_list();
		} else if (test_include()) {
			handle_include();
		} else {
			throw std::runtime_error(std::string(LT(1).begin()) + ": Error: expected declaration or include");
		}
	}

	return std::shared_ptr<Program>(new Program(stmts));
}

std::shared_ptr<Stmt> Parser::stmt() {
	if (test_test()) {
		return test();
	} else if (LA(1) == Token::category::macro) {
		return macro();
	} else if (LA(1) == Token::category::param) {
		return param();
	} else if (test_controller()) {
		return controller();
	} else if (test_macro_call()) {
		return macro_call<AST::Stmt>();
	} else {
		throw std::runtime_error(std::string(LT(1).begin())
			+ ": Error: unsupported statement: " + LT(1).value());
	}
}

std::shared_ptr<AST::StmtBlock> Parser::stmt_block() {
	Token lbrace = LT(1);
	match(Token::category::lbrace);

	newline_list();
	std::vector<std::shared_ptr<Stmt>> stmts;

	while (test_stmt()) {
		auto st = stmt();
		stmts.push_back(st);
		newline_list();
	}

	Token rbrace = LT(1);
	match(Token::category::rbrace);

	auto statement = std::shared_ptr<StmtBlock>(new StmtBlock(lbrace, rbrace,  stmts));
	return statement;
}

std::shared_ptr<Test> Parser::test() {
	std::shared_ptr<AttrBlock> attrs(nullptr);
	//To be honest, we should place attr list in a separate Node. And we will do that
	//just when it could be used somewhere else
	if (LA(1) == Token::category::lbracket) {
		attrs = attr_block("test_global");
		newline_list();
	}

	Token test = LT(1);
	match(Token::category::test);

	std::shared_ptr<StringTokenUnion> name = string_token_union(Token::category::id);

	std::vector<std::shared_ptr<StringTokenUnion>> parents;

	if (LA(1) == Token::category::colon) {
 		match(Token::category::colon);
 		newline_list();
 		parents.push_back(string_token_union(Token::category::id));

 		while (LA(1) == Token::category::comma) {
 			match(Token::category::comma);
 			newline_list();
 			parents.push_back(string_token_union(Token::category::id));
 		}
	}

	newline_list();
	auto commands = command_block();
	return std::make_shared<Test>(attrs, test, name, parents, commands);
}

std::shared_ptr<MacroArg> Parser::macro_arg() {
	Token arg_name = LT(1);
	match(Token::category::id);

	std::shared_ptr<String> default_value = nullptr;

	if (LA(1) == Token::category::assign) {
		match(Token::category::assign);

		default_value = string();
	}

	return std::shared_ptr<MacroArg>(new MacroArg(arg_name, default_value));
}

std::vector<Token> Parser::macro_body(const std::string& name) {
	std::vector<Token> result;

	result.push_back(LT(1));
	match(Token::category::lbrace);

	size_t braces_count = 1;

	while (braces_count != 0) {
		if (LA(1) == Token::category::lbrace) {
			braces_count++;
		} else if (LA(1) == Token::category::rbrace) {
			braces_count--;
		} else if (LA(1) == Token::category::eof) {
			throw std::runtime_error(std::string(LT(1).begin()) + ": Error: macro \"" + name + "\" body reached the end of file without closing \"}\"");
		}

		result.push_back(LT(1));
		match(LA(1));
	}

	return result;
}

std::shared_ptr<Macro> Parser::macro() {
	Token macro = LT(1);
	match(Token::category::macro);

	Token name = LT(1);
	match(Token::category::id);
	match(Token::category::lparen);

	std::vector<std::shared_ptr<MacroArg>> args;

	if (LA(1) == Token::category::id) {
		args.push_back(macro_arg());
	}

	while (LA(1) == Token::category::comma) {
		if (args.empty()) {
			match(Token::category::rparen); //will cause failure
		}
		match(Token::category::comma);
		args.push_back(macro_arg());
	}

	match(Token::category::rparen);

	newline_list();
	auto body = macro_body(name.value());

	return std::make_shared<Macro>(macro, name, args, body);
}

std::shared_ptr<Param> Parser::param() {
	Token param_token = LT(1);
	match(Token::category::param);

	Token name = LT(1);
	match(Token::category::id);

	auto value = string();

	return std::make_shared<Param>(param_token, name, value);
}

std::shared_ptr<Attr> Parser::attr(const std::string& ctx_name) {
	Token name = LT(1);

	match(Token::category::id);

	Token id = Token();

	if (LA(1) == Token::category::id) {
		id = LT(1);
		match(Token::category::id);
	}

	match(Token::category::colon);
	newline_list();

	std::shared_ptr<IAttrValue> value;
	if (LA(1) == Token::category::lbrace) {
		auto block = attr_block(name.value());
		value = std::make_shared<AttrValue<AttrBlock>>(block);
	} else {
		//string token union
		//Let's check it's either string or number or size or boolean

		if (!test_string() &&
			(LA(1) != Token::category::number) &&
			(LA(1) != Token::category::size) &&
			(LA(1) != Token::category::boolean))
		{
			throw std::runtime_error(std::string(LT(1).begin()) + ": Unknown attr type: " + LT(1).value());
		}

		if (LA(1) == Token::category::triple_quoted_string) {
			throw std::runtime_error(std::string(LT(1).begin()) + ": Can't accept multiline as an attr value: " + LT(1).value());
		}

		auto simple_value = string_token_union(LA(1));
		auto simple_attr = std::make_shared<SimpleAttr>(simple_value);
		value = std::make_shared<AttrValue<SimpleAttr>>(simple_attr);
	}

	return std::make_shared<Attr>(name, id, value);
}

std::shared_ptr<AttrBlock> Parser::attr_block(const std::string& ctx_name) {
	Token lbrace = LT(1);

	match({Token::category::lbrace, Token::category::lbracket});

	newline_list();
	std::vector<std::shared_ptr<Attr>> attrs;

	while (LA(1) == Token::category::id) {
		attrs.push_back(attr(ctx_name));
		if ((LA(1) == Token::category::rbrace) || (LA(1) == Token::category::rbracket)) {
			break;
		}
		match(Token::category::newline);
		newline_list();
	}

	newline_list();
	Token rbrace = LT(1);
	if (lbrace.type() == Token::category::lbrace) {
		match(Token::category::rbrace);
	} else {
		match (Token::category::rbracket);
	}

	return std::shared_ptr<AttrBlock>(new AttrBlock(lbrace, rbrace, attrs));
}

std::shared_ptr<AST::Controller> Parser::controller() {
	Token controller = LT(1);

	match ({Token::category::machine, Token::category::flash, Token::category::network});

	auto name = string_token_union(Token::category::id);

	newline_list();
	if (LA(1) != Token::category::lbrace) {
		throw std::runtime_error(std::string(LT(1).begin()) + ":Error: expected attribute block");
	}

	std::string ctx_name;
	if (controller.type() == Token::category::machine) {
		ctx_name = "vm_global";
	} else if (controller.type() == Token::category::flash) {
		ctx_name = "fd_global";
	} else if (controller.type() == Token::category::network) {
		ctx_name = "network_global";
	} else {
		throw std::runtime_error("Should never happen");
	}
	auto block = attr_block(ctx_name);
	return std::make_shared<AST::Controller>(controller, name, block);
}

std::shared_ptr<Cmd> Parser::command() {
	if (test_macro_call()) {
		return macro_call<AST::Cmd>();
	} else {
		auto entity = string_token_union(Token::category::id);
		std::shared_ptr<Action> act = action();
		return std::make_shared<AST::RegularCmd>(entity, act);
	}
}

std::shared_ptr<CmdBlock> Parser::command_block() {
	Token lbrace = LT(1);
	match(Token::category::lbrace);

	newline_list();
	std::vector<std::shared_ptr<Cmd>> commands;

	while (test_command()) {
		commands.push_back(command());
		newline_list();
	}

	Token rbrace = LT(1);
	match(Token::category::rbrace);

	return std::shared_ptr<CmdBlock>(new CmdBlock(lbrace, rbrace, commands));
}

std::shared_ptr<KeyCombination> Parser::key_combination() {
	std::vector<Token> buttons;

	do {
		buttons.push_back(LT(1));
		match(Token::category::id);

		if (LA(1) == Token::category::plus) {
			match(Token::category::plus);
		}
	} while (LA(1) == Token::category::id);

	return std::shared_ptr<KeyCombination>(new KeyCombination(buttons));
}

std::shared_ptr<KeySpec> Parser::key_spec() {
	auto combination = key_combination();

	std::shared_ptr<StringTokenUnion> times = nullptr;

	if (LA(1) == Token::category::asterisk) {
		match(Token::category::asterisk);
		times = string_token_union(Token::category::number);
	}

	return std::shared_ptr<KeySpec>(new KeySpec(combination, times));
}

std::shared_ptr<Action> Parser::action() {
	std::shared_ptr<Action> action;
	if (LA(1) == Token::category::abort) {
		action = abort();
	} else if (LA(1) == Token::category::print) {
		action = print();
	} else if (LA(1) == Token::category::type_) {
		action = type();
	} else if (LA(1) == Token::category::wait) {
		action = wait();
	} else if (LA(1) == Token::category::sleep) {
		action = sleep();
	} else if (LA(1) == Token::category::press) {
		action = press();
	} else if (LA(1) == Token::category::hold) {
		action = hold();
	} else if (LA(1) == Token::category::release) {
		action = release();
	} else if (LA(1) == Token::category::mouse) {
		action = mouse();
	} else if ((LA(1) == Token::category::plug) || (LA(1) == Token::category::unplug)) {
		action = plug();
	} else if (LA(1) == Token::category::start) {
		action = start();
	} else if (LA(1) == Token::category::stop) {
		action = stop();
	} else if (LA(1) == Token::category::shutdown) {
		action = shutdown();
	} else if (LA(1) == Token::category::exec) {
		action = exec();
	} else if ((LA(1) == Token::category::copyto) || (LA(1) == Token::category::copyfrom)) {
		action = copy();
	} else if (LA(1) == Token::category::screenshot) {
		action = screenshot();
	} else if (LA(1) == Token::category::lbrace) {
		action = action_block();
	} else if (LA(1) == Token::category::if_) {
		action = if_clause();
	} else if (LA(1) == Token::category::for_) {
		action = for_clause();
	} else if ((LA(1) == Token::category::break_) || (LA(1) == Token::category::continue_)) {
		action = cycle_control();
	} else if (LA(1) == Token::category::semi || LA(1) == Token::category::newline) {
		return empty_action();
	} else if (test_macro_call()) {
		action = macro_call<AST::Action>();
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: Unknown action: " + LT(1).value());
	}

	if (action->t.type() != Token::category::action_block &&
		action->t.type() != Token::category::if_ &&
		action->t.type() != Token::category::for_)
	{
		Token delim;
		if (LA(1) == Token::category::newline) {
			delim = LT(1);
			match(Token::category::newline);
		} else if (LA(1) == Token::category::semi) {
			delim = LT(1);
			match(Token::category::semi);
		} else {
			throw std::runtime_error(std::string(LT(1).begin()) +
				": Expected new line or ';'");
		}
		action->delim = delim;
	}

	return action;
}

std::shared_ptr<Empty> Parser::empty_action() {
	match({Token::category::semi, Token::category::newline});
	return std::make_shared<Empty>();
}

std::shared_ptr<Abort> Parser::abort() {
	Token abort_token = LT(1);
	match(Token::category::abort);

	Token value = LT(1);

	auto message = string();

	return std::make_shared<Abort>(abort_token, message);
}

std::shared_ptr<Print> Parser::print() {
	Token print_token = LT(1);
	match(Token::category::print);

	Token value = LT(1);

	auto message = string();

	return std::make_shared<Print>(print_token, message);
}

std::shared_ptr<Type> Parser::type() {
	Token type_token = LT(1);
	match(Token::category::type_);

	Token value = LT(1);
	auto text = string();

	std::shared_ptr<StringTokenUnion> interval = nullptr;

	if (test_string() || LA(1) == Token::category::interval) {
		match(Token::category::interval);
		interval = string_token_union(Token::category::time_interval);
	}
	return std::make_shared<Type>(type_token, text, interval);
}

std::shared_ptr<Wait> Parser::wait() {
	Token wait_token = LT(1);
	match(Token::category::wait);

	std::shared_ptr<SelectExpr> select_expression(nullptr);
	std::shared_ptr<StringTokenUnion> timeout = nullptr;
	std::shared_ptr<StringTokenUnion> interval = nullptr;

	if (!test_selectable()) {
		throw std::runtime_error(std::string(LT(1).begin()) + " : Error: expexted an object to wait");
	}

	select_expression = select_expr();

	//special check for multiline strings. We don't support them yet.

	//ToDo: Thiple check this part
	if (select_expression->t.type() == Token::category::triple_quoted_string) {
		throw std::runtime_error(std::string(select_expression->begin()) +
			": Error: multiline strings are not supported in wait action");
	}

	if (LA(1) == Token::category::timeout) {
		match(Token::category::timeout);
		timeout = string_token_union(Token::category::time_interval);
	}

	if (LA(1) == Token::category::interval) {
		match(Token::category::interval);
		interval = string_token_union(Token::category::time_interval);
	}

	return std::make_shared<Wait>(wait_token, select_expression, timeout, interval);
}

std::shared_ptr<AST::Sleep> Parser::sleep() {
	Token sleep_token = LT(1);
	match(Token::category::sleep);

	auto timeout = string_token_union(Token::category::time_interval);

	return std::make_shared<AST::Sleep>(sleep_token, timeout);
}

std::shared_ptr<Press> Parser::press() {
	Token press_token = LT(1);
	match(Token::category::press);

	std::vector<std::shared_ptr<KeySpec>> keys;
	keys.push_back(key_spec());

	while (LA(1) == Token::category::comma) {
		match(Token::category::comma);
		keys.push_back(key_spec());
	}

	std::shared_ptr<StringTokenUnion> interval = nullptr;

	if (LA(1) == Token::category::interval) {
		match (Token::category::interval);
		interval = string_token_union(Token::category::time_interval);
	}

	return std::make_shared<Press>(press_token, keys, interval);
}

std::shared_ptr<Hold> Parser::hold() {
	Token hold_token = LT(1);
	match(Token::category::hold);

	auto combination = key_combination();

	return std::make_shared<Hold>(hold_token, combination);
}

std::shared_ptr<Release> Parser::release() {
	Token release_token = LT(1);
	match(Token::category::release);

	std::shared_ptr<AST::KeyCombination> combination = nullptr;
	if (LA(1) == Token::category::id) {
		combination = key_combination();
	}

	return std::make_shared<Release>(release_token, combination);
}

std::shared_ptr<AST::Mouse> Parser::mouse() {
	Token mouse_token = LT(1);
	match(Token::category::mouse);

	std::shared_ptr<MouseEvent> event = nullptr;

	if (LA(1) == Token::category::move ||
		LA(1) == Token::category::click ||
		LA(1) == Token::category::lclick ||
		LA(1) == Token::category::rclick ||
		LA(1) == Token::category::mclick ||
		LA(1) == Token::category::dclick)
	{
		event = mouse_move_click();
	} else if (LA(1) == Token::category::hold) {
		event = mouse_hold();
	} else if (LA(1) == Token::category::release) {
		event = mouse_release();
	} else if (LA(1) == Token::category::wheel) {
		event = mouse_wheel();
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + " : Error: unknown mouse action: " + LT(1).value());
	}

	return std::make_shared<Mouse>(mouse_token, event);
}

std::shared_ptr<MouseAdditionalSpecifier> Parser::mouse_additional_specifier() {
	Token tmp = LT(1);
	match(Token::category::dot);
	Token name = LT(1);
	if (!Pos::is_adjacent(tmp.end(), name.begin())) {
		throw std::runtime_error(std::string(tmp.end()) + ": Error: expected a mouse specifier name");
	}
	match(Token::category::id);
	Token lparen = LT(1);
	match(Token::category::lparen);

	Token arg;
	if (LA(1) != Token::category::rparen && LA(1) != Token::category::number) {
		throw std::runtime_error(std::string(LT(1).begin()) + " : Error: you can use only numbers as arguments in cursor specifiers");
	}

	if (LA(1) == Token::category::number) {
		arg = LT(1);
		match(Token::category::number);
	}

	Token rparen = LT(1);
	match(Token::category::rparen);

	return std::make_shared<MouseAdditionalSpecifier>(name, lparen, arg, rparen);
}

std::shared_ptr<MouseSelectable> Parser::mouse_selectable() {
	auto select = selectable();

	std::vector<std::shared_ptr<MouseAdditionalSpecifier>> specifiers;

	auto select_end = select->end();
	auto tmp = LT(1);

	for (Pos it = select->end(); LA(1) == Token::category::dot && Pos::is_adjacent(it, LT(1).begin());) {
		auto specifier = mouse_additional_specifier();
		specifiers.push_back(specifier);
		it = specifier->end();
	}

	std::shared_ptr<StringTokenUnion> timeout = nullptr;

	if (LA(1) == Token::category::timeout) {
		match(Token::category::timeout);
		timeout = string_token_union(Token::category::time_interval);
	}

	return std::make_shared<MouseSelectable>(select, specifiers, timeout);
}

std::shared_ptr<AST::MouseMoveClick> Parser::mouse_move_click() {
	Token event_token = LT(1);
	match({Token::category::click,
		Token::category::lclick,
		Token::category::move,
		Token::category::rclick,
		Token::category::mclick,
		Token::category::dclick});

	std::shared_ptr<MouseMoveTarget> target = nullptr;

	if (test_selectable()) {
		target = mouse_selectable();
	} else if (LA(1) == Token::category::number) {
		target = mouse_coordinates();
	}

	if (event_token.type() == Token::category::move && !target) {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: you must specify a target to move the mouse cursor");
	}

	return std::make_shared<MouseMoveClick>(event_token, target);
}

std::shared_ptr<AST::MouseHold> Parser::mouse_hold() {
	Token event_token = LT(1);
	match(Token::category::hold);

	Token button = LT(1);
	match({Token::category::lbtn, Token::category::rbtn, Token::category::mbtn});

	return std::make_shared<MouseHold>(event_token, button);
}

std::shared_ptr<AST::MouseRelease> Parser::mouse_release() {
	Token event_token = LT(1);
	match(Token::category::release);

	return std::make_shared<MouseRelease>(event_token);
}

std::shared_ptr<AST::MouseWheel> Parser::mouse_wheel() {
	Token event_token = LT(1);
	match(Token::category::wheel);

	Token direction = LT(1);

	if (direction.value() != "up" && direction.value() != "down") {
		throw std::runtime_error(std::string(direction.begin()) + " : Error: unknown wheel direction: " + direction.value());
	}

	match(Token::category::id);

	return std::make_shared<MouseWheel>(event_token, direction);
}

std::shared_ptr<MouseCoordinates> Parser::mouse_coordinates() {
	auto dx = LT(1);
	match(Token::category::number);
	auto dy = LT(1);
	match(Token::category::number);

	return std::make_shared<MouseCoordinates>(dx, dy);
}

std::shared_ptr<AST::PlugResource> Parser::plug_resource() {
	std::shared_ptr<PlugResource> result = nullptr;
	if (LA(1) == Token::category::flash) {
		result = plug_resource_flash();
	} else if (LA(1) == Token::category::dvd) {
		result = plug_resource_dvd();
	} else if (LA(1) == Token::category::hostdev) {
		result = plug_resource_hostdev();
	} else if (LT(1).value() == "nic") {
		result = plug_resource_nic();
	} else if (LT(1).value() == "link") {
		result = plug_resource_link();
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: Unknown device type for plug/unplug: " + LT(1).value());
	}

	return result;
}

std::shared_ptr<AST::PlugFlash> Parser::plug_resource_flash() {
	Token flash_token = LT(1);
	match(Token::category::flash);

	auto name = string_token_union(Token::category::id);

	return std::make_shared<AST::PlugFlash>(flash_token, name);
}

std::shared_ptr<AST::PlugNIC> Parser::plug_resource_nic() {
	Token nic_token = LT(1);
	match(Token::category::id);

	auto name = string_token_union(Token::category::id);

	return std::make_shared<AST::PlugNIC>(nic_token, name);
}

std::shared_ptr<AST::PlugLink> Parser::plug_resource_link() {
	Token link_token = LT(1);
	match(Token::category::id);

	auto name = string_token_union(Token::category::id);

	return std::make_shared<AST::PlugLink>(link_token, name);
}

std::shared_ptr<AST::PlugDVD> Parser::plug_resource_dvd() {
	Token dvd_token = LT(1);
	match(Token::category::dvd);

	std::shared_ptr<AST::String> path = nullptr;

	if (test_string()) {
		path = string();
	}

	return std::make_shared<AST::PlugDVD>(dvd_token, path);
}

std::shared_ptr<AST::PlugHostDev> Parser::plug_resource_hostdev() {
	Token hostdev_token = LT(1);
	match(Token::category::hostdev);

	if (LA(1) != Token::category::usb) {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: Unknown usb device type for plug/unplug: " + LT(1).value());
	}

	Token type = LT(1);

	match(Token::category::usb);
	std::shared_ptr<AST::String> addr = string();

	return std::make_shared<AST::PlugHostDev>(hostdev_token, type, addr);
}

std::shared_ptr<Plug> Parser::plug() {
	Token plug_token = LT(1);

	if (LA(1) == Token::category::plug) {
		match(Token::category::plug);
	} else {
		match(Token::category::unplug);
	}

	auto resource = plug_resource();
	return std::make_shared<Plug>(plug_token, resource);
}

std::shared_ptr<Start> Parser::start() {
	Token start_token = LT(1);
	match(Token::category::start);

	return std::make_shared<Start>(start_token);
}

std::shared_ptr<Stop> Parser::stop() {
	Token stop_token = LT(1);
	match(Token::category::stop);

	return std::make_shared<Stop>(stop_token);
}

std::shared_ptr<Shutdown> Parser::shutdown() {
	Token shutdown_token = LT(1);
	match(Token::category::shutdown);

	std::shared_ptr<StringTokenUnion> timeout = nullptr;

	if (LA(1) == Token::category::timeout) {
		match(Token::category::timeout);
		timeout = string_token_union(Token::category::time_interval);
	}

	return std::make_shared<Shutdown>(shutdown_token, timeout);
}

std::shared_ptr<Exec> Parser::exec() {
	Token exec_token = LT(1);
	match(Token::category::exec);

	Token process_token = LT(1);
	match(Token::category::id);

	auto commands = string();

	std::shared_ptr<StringTokenUnion> timeout = nullptr;

	if (LA(1) == Token::category::timeout) {
		match(Token::category::timeout);
		timeout = string_token_union(Token::category::time_interval);
	}

	return std::make_shared<Exec>(exec_token, process_token, commands, timeout);
}

std::shared_ptr<Copy> Parser::copy() {
	Token copy_token = LT(1);
	match({Token::category::copyto, Token::category::copyfrom});

	auto from = string();
	auto to = string();

	Token nocheck = Token();

	if (LT(1).value() == "nocheck") {
		nocheck = LT(1);
		match(Token::category::id);
	}

	std::shared_ptr<StringTokenUnion> timeout = nullptr;

	if (LA(1) == Token::category::timeout) {
		match(Token::category::timeout);
		timeout = string_token_union(Token::category::time_interval);
	}

	return std::make_shared<Copy>(copy_token, from, to, nocheck, timeout);
}

std::shared_ptr<Screenshot> Parser::screenshot() {
	Token screenshot_token = LT(1);
	match (Token::category::screenshot);

	auto destination = string();

	return std::make_shared<Screenshot>(screenshot_token, destination);
}

std::shared_ptr<ActionBlock> Parser::action_block() {
	Token lbrace = LT(1);
	match(Token::category::lbrace);

	newline_list();
	std::vector<std::shared_ptr<Action>> actions;

	while (test_action()) {
		auto act = action();
		actions.push_back(act);
		newline_list();
	}

	Token rbrace = LT(1);
	match(Token::category::rbrace);

	return std::make_shared<ActionBlock>(lbrace, rbrace,  actions);
}

template <typename BaseType>
std::shared_ptr<MacroCall<BaseType>> Parser::macro_call() {
	Token macro_name = LT(1);
	match(Token::category::id);

	match(Token::category::lparen);

	std::vector<std::shared_ptr<String>> params;

	if (test_string()) {
		params.push_back(string());
	}

	while (LA(1) == Token::category::comma) {
		if (params.empty()) {
			match(Token::category::rparen); //will cause failure
		}
		match(Token::category::comma);
		params.push_back(string());
	}

	match(Token::category::rparen);
	return std::make_shared<MacroCall<BaseType>>(macro_name, params);
}

std::shared_ptr<IfClause> Parser::if_clause() {
	Token if_token = LT(1);
	match(Token::category::if_);

	Token open_paren = LT(1);
	match(Token::category::lparen);

	auto expression = expr();
	Token close_paren = LT(1);
	match(Token::category::rparen);

	newline_list();

	auto if_action = action();

	newline_list();
	Token else_token = Token();
	std::shared_ptr<Action> else_action = nullptr;

	if (LA(1) == Token::category::else_) {
		else_token = LT(1);
		match(Token::category::else_);
		newline_list();
		else_action = action();
	}

	return std::make_shared<IfClause>(
		if_token,
		open_paren, expression,
		close_paren, if_action,
		else_token, else_action
	);
}

std::shared_ptr<Range> Parser::range() {
	Token range_token = LT(1);
	match(Token::category::RANGE);

	std::shared_ptr<StringTokenUnion> r1 = string_token_union(Token::category::number);
	std::shared_ptr<StringTokenUnion> r2 = nullptr;
	if (test_string() || LA(1) == Token::category::number) {
		r2 = string_token_union(Token::category::number);
	}

	return std::make_shared<Range>(range_token, r1, r2);
}

std::shared_ptr<CounterList> Parser::counter_list() {
	if (LA(1) == Token::category::RANGE) {
		return range();
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: Unknown counter_list specifier: " + LT(1).value());
	}
}

std::shared_ptr<ForClause> Parser::for_clause() {
	Token for_token = LT(1);
	match(Token::category::for_);

	match(Token::category::lparen);
	Token counter = LT(1);

	match(Token::category::id);

	match(Token::category::IN_);

	if (!test_counter_list()) {
		throw std::runtime_error(std::string(LT(1).begin()) + " : Error: expexted a RANGE");
	}

	std::shared_ptr<CounterList> list = counter_list();
	match(Token::category::rparen);
	newline_list();

	auto cycle_body = action();

	Token else_token = Token();
	std::shared_ptr<Action> else_action = nullptr;

	if (LA(1) == Token::category::else_) {
		else_token = LT(1);
		match(Token::category::else_);
		newline_list();
		else_action = action();
	}

	return std::make_shared<ForClause>(
		for_token,
		counter,
		list,
		cycle_body,
		else_token,
		else_action
	);
}

std::shared_ptr<CycleControl> Parser::cycle_control() {
	Token control_token = LT(1);
	match({Token::category::break_, Token::category::continue_});

	return std::make_shared<CycleControl>(control_token);
}

std::shared_ptr<SelectExpr> Parser::select_expr() {
	auto left = selectable();
	if ((LA(1) == Token::category::double_ampersand) ||
		(LA(1) == Token::category::double_vertical_bar)) {
		return select_binop(left);
	} else {
		return left;
	}
}

std::shared_ptr<AST::SelectParentedExpr> Parser::select_parented_expr() {
	auto lparen = LT(1);
	match(Token::category::lparen);

	auto expression = select_expr();

	auto rparen = LT(1);
	match(Token::category::rparen);
	return std::shared_ptr<AST::SelectParentedExpr>(new AST::SelectParentedExpr(lparen, expression, rparen));
}

std::shared_ptr<SelectBinOp> Parser::select_binop(std::shared_ptr<SelectExpr> left) {
	auto op = LT(1);

	match({Token::category::double_ampersand, Token::category::double_vertical_bar});
	newline_list();

	auto right = select_expr();

	return std::make_shared<AST::SelectBinOp>(left, op, right);
}

std::shared_ptr<Selectable> Parser::selectable() {
	auto not_token = Token();
	if (LA(1) == Token::category::exclamation_mark) {
		not_token = LT(1);
		match(Token::category::exclamation_mark);
	}

	std::shared_ptr<Selectable> selectable = nullptr;

	if (test_string()) {
		selectable = select_text();
	} else if(LA(1) == Token::category::js) {
		selectable = select_js();
	} else if(LA(1) == Token::category::img) {
		selectable = select_img();
	} else if(LA(1) == Token::category::homm3) {
		selectable = select_homm3();
	} else if(LA(1) == Token::category::lparen) {
		selectable = select_parented_expr();
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + ":Error: Unknown selective object type: " + LT(1).value());
	}

	selectable->excl_mark = not_token;
	return selectable;
}

std::shared_ptr<SelectJS> Parser::select_js() {
	Token js = LT(1);
	match(Token::category::js);
	auto script = string();
	return std::shared_ptr<SelectJS>(new SelectJS(js, script));
}

std::shared_ptr<SelectImg> Parser::select_img() {
	Token img = LT(1);
	match(Token::category::img);
	auto img_path = string();
	return std::shared_ptr<SelectImg>(new SelectImg(img, img_path));
}

std::shared_ptr<SelectHomm3> Parser::select_homm3() {
	Token homm3 = LT(1);
	match(Token::category::homm3);
	auto id = string();
	return std::shared_ptr<SelectHomm3>(new SelectHomm3(homm3, id));
}

std::shared_ptr<SelectText> Parser::select_text() {
	auto text = string();
	return std::shared_ptr<SelectText>(new SelectText(text));
}

std::shared_ptr<String> Parser::string() {
	Token str = LT(1);
	if (!test_string()) {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: expected string");
	}

	match({Token::category::quoted_string, Token::category::triple_quoted_string});

	auto new_node = std::make_shared<String>(str);

	try {
		template_literals::Parser templ_parser;
		templ_parser.validate_sanity(new_node->text());
	} catch (const std::runtime_error& error) {
		std::throw_with_nested(std::runtime_error(std::string(new_node->begin()) + ": Error parsing string: \"" + new_node->text() + "\""));
	}

	return new_node;
}

std::shared_ptr<AST::StringTokenUnion> Parser::string_token_union(Token::category expected_token_type) {
	if (!test_string() && LA(1) != expected_token_type) {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: expected a string or " + Token::type_to_string(expected_token_type) + ", but got " +
			Token::type_to_string(LA(1)) + " " + LT(1).value());
	}

	Token token;
	std::shared_ptr<String> str = nullptr;

	if (test_string()) {
		str = string();
	} else {
		token = LT(1);
		match(expected_token_type);
	}

	return std::shared_ptr<StringTokenUnion>(new StringTokenUnion(token, str, expected_token_type));
}

std::shared_ptr<ParentedExpr> Parser::parented_expr() {
	auto lparen = LT(1);
	match(Token::category::lparen);

	auto expression = expr();

	auto rparen = LT(1);
	match(Token::category::rparen);
	return std::shared_ptr<AST::ParentedExpr>(new AST::ParentedExpr(lparen, expression, rparen));
}

std::shared_ptr<Comparison> Parser::comparison() {
	auto left = string();

	Token op = LT(1);

	match({
		Token::category::GREATER,
		Token::category::LESS,
		Token::category::EQUAL,
		Token::category::STRGREATER,
		Token::category::STRLESS,
		Token::category::STREQUAL
		});

	auto right = string();

	return std::shared_ptr<Comparison>(new Comparison(op, left, right));
}

std::shared_ptr<Defined> Parser::defined() {
	auto defined_token = LT(1);
	match(Token::category::DEFINED);
	Token var = LT(1);
	match(Token::category::id);

	return std::shared_ptr<Defined>(new Defined(defined_token, var));
}

std::shared_ptr<Check> Parser::check() {
	Token check_token = LT(1);
	match(Token::category::check);

	std::shared_ptr<SelectExpr> select_expression(nullptr);

	if (!test_selectable()) {
		throw std::runtime_error(std::string(LT(1).begin()) + " : Error: expexted an object to check");
	}

	select_expression = select_expr();

	std::shared_ptr<StringTokenUnion> timeout = nullptr;
	std::shared_ptr<StringTokenUnion> interval = nullptr;

	if (LA(1) == Token::category::timeout) {
		match(Token::category::timeout);
		timeout = string_token_union(Token::category::time_interval);
	}

	if (LA(1) == Token::category::interval) {
		match(Token::category::interval);
		interval = string_token_union(Token::category::time_interval);
	}

	return std::make_shared<Check>(check_token, select_expression, timeout, interval);
}

std::shared_ptr<BinOp> Parser::binop(std::shared_ptr<Expr> left) {
	auto op = LT(1);

	match({Token::category::OR, Token::category::AND});
	newline_list();

	auto right = expr();

	return std::make_shared<BinOp>(op, left, right);
}

std::shared_ptr<Negation> Parser::negation() {
	auto not_token = LT(1);
	match(Token::category::NOT);

	return std::make_shared<Negation>(not_token, expr());
}

std::shared_ptr<Expr> Parser::expr() {
	std::shared_ptr<Expr> left = nullptr;

	if (LA(1) == Token::category::NOT) {
		left = negation();
	} else if (LA(1) == Token::category::check) {
		left = check();
	} else if(test_comparison()) {
		left = comparison();
	} else if(test_defined()) {
		left = defined();
	} else if (LA(1) == Token::category::lparen) {
		left = parented_expr();
	} else if (test_string()) {
		left = std::make_shared<StringExpr>(string());
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: Unknown expression: " + LT(1).value());
	}

	if ((LA(1) == Token::category::AND) ||
		(LA(1) == Token::category::OR)) {
		return binop(left);
	} else {
		return left;
	}
}

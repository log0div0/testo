
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
	for (int i = 0; i < 2; i++) {
		consume();	//Populate lookahead buffer with tokens
	}
}

void Parser::consume() {
	Ctx& current_lexer = lexers[lexers.size() - 1];

	current_lexer.lookahead[current_lexer.p] = current_lexer.lex.get_next_token();
	current_lexer.p = (current_lexer.p + 1) % 2;
}

void Parser::match(Token::category type) {
	if (LA(1) == type) {
		consume();
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) +
			": unexpected token \"" +
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
			": unexpected token \"" +
			LT(1).value() + "\""); //TODO: more informative what we expected
}

Token Parser::LT(size_t i) const {
	return lexers[lexers.size() - 1].lookahead[(lexers[lexers.size() - 1].p + i - 1) % 2]; //circular fetch
}

Token::category Parser::LA(size_t i) const {
	return LT(i).type();
}

bool Parser::test_stmt() const {
	return ((LA(1) == Token::category::macro) ||
		(LA(1) == Token::category::param) ||
		test_controller() ||
		test_test());
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

bool Parser::test_command() const {
	return (LA(1) == Token::category::id);
}

bool Parser::test_action() const {
	return ((LA(1) == Token::category::abort) ||
		(LA(1) == Token::category::print) ||
		(LA(1) == Token::category::type_) ||
		(LA(1) == Token::category::wait) ||
		(LA(1) == Token::category::sleep) ||
		(LA(1) == Token::category::press) ||
		(LA(1) == Token::category::hold) ||
		(LA(1) == Token::category::release) ||
		(LA(1) == Token::category::mouse) ||
		(LA(1) == Token::category::plug) ||
		(LA(1) == Token::category::unplug) ||
		(LA(1) == Token::category::start) ||
		(LA(1) == Token::category::stop) ||
		(LA(1) == Token::category::shutdown) ||
		(LA(1) == Token::category::exec) ||
		(LA(1) == Token::category::copyto) ||
		(LA(1) == Token::category::copyfrom) ||
		(LA(1) == Token::category::lbrace) ||
		(LA(1) == Token::category::if_) ||
		(LA(1) == Token::category::for_) ||
		(LA(1) == Token::category::break_) ||
		(LA(1) == Token::category::continue_) ||
		(LA(1) == Token::category::semi) ||
		(LA(1) == Token::category::id)); //macro call
}

bool Parser::test_counter_list() const {
	return LA(1) == Token::category::RANGE;
}

bool Parser::test_string() const {
	return ((LA(1) == Token::category::quoted_string) ||
		(LA(1) == Token::category::triple_quoted_string));
}

bool Parser::test_selectable() const {
	return (test_string() || (LA(1) == Token::category::js)  ||
		(LA(1) == Token::category::exclamation_mark) ||
		(LA(1) == Token::category::lparen));
}

bool Parser::test_binary() const {
	return ((LA(1) == Token::category::true_) ||
		(LA(1) == Token::category::false_));
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
		auto current_path = lexers[lexers.size() - 1].lex.file();
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

	for (int i = 0; i < 2; i++) {
		consume();	//Populate lookahead buffer with tokens
	}
}

std::shared_ptr<Program> Parser::parse() {
	std::vector<std::shared_ptr<IStmt>> stmts;

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

std::shared_ptr<IStmt> Parser::stmt() {
	if (test_test()) {
		return test();
	} else if (LA(1) == Token::category::macro) {
		return macro();
	} else if (LA(1) == Token::category::param) {
		return param();
	} else if (test_controller()) {
		return controller();
	} else {
		throw std::runtime_error(std::string(LT(1).begin())
			+ ": Error: unsupported statement: " + LT(1).value());
	}
}

std::shared_ptr<Stmt<Test>> Parser::test() {
	std::shared_ptr<AttrBlock> attrs(nullptr);
	//To be honest, we should place attr list in a separate Node. And we will do that
	//just when it could be used somewhere else
	if (LA(1) == Token::category::lbracket) {
		attrs = attr_block();
		newline_list();
	}

	Token test = LT(1);
	match(Token::category::test);

	Token name = LT(1);

	match(Token::category::id);

	std::vector<Token> parents;

	if (LA(1) == Token::category::colon) {
 		match(Token::category::colon);
 		newline_list();
 		parents.push_back(LT(1));
 		match(Token::category::id);

 		while (LA(1) == Token::category::comma) {
 			match(Token::category::comma);
 			newline_list();
 			parents.push_back(LT(1));
 			match(Token::category::id);
 		}
	}

	newline_list();
	auto commands = command_block();
	auto stmt = std::shared_ptr<Test>(new Test(attrs, test, name, parents, commands));

	return std::shared_ptr<Stmt<Test>>(new Stmt<Test>(stmt));
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

std::shared_ptr<Stmt<Macro>> Parser::macro() {
	Token macro = LT(1);
	match(Token::category::macro);

	Token name = LT(1);
	match(Token::category::id);

	if (LA(1) != Token::category::lparen) {
		throw std::runtime_error(std::string(name.begin()) + ": Error: unknown action: " + name.value());
	}

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
	auto actions = action_block();

	auto stmt = std::shared_ptr<Macro>(new Macro(macro, name, args, actions));

	return std::shared_ptr<Stmt<Macro>>(new Stmt<Macro>(stmt));
}

std::shared_ptr<Stmt<Param>> Parser::param() {
	Token param_token = LT(1);
	match(Token::category::param);

	Token name = LT(1);
	match(Token::category::id);

	auto value = string();

	auto stmt = std::shared_ptr<Param>(new Param(param_token, name, value));

	return std::shared_ptr<Stmt<Param>>(new Stmt<Param>(stmt));
}

std::shared_ptr<Attr> Parser::attr() {
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
		auto block = attr_block();
		value = std::shared_ptr<AttrValue<AttrBlock>>(new AttrValue<AttrBlock>(block));
	} else if (test_string()) {
		auto str = string();
		if (str->t.type() == Token::category::triple_quoted_string) {
			throw std::runtime_error(std::string(str->begin()) + ": Can't accept multiline as an attr value: " + std::string(*str));
		}
		auto string_value = std::shared_ptr<StringAttr>(new StringAttr(str));
		value = std::shared_ptr<AttrValue<StringAttr>>(new AttrValue<StringAttr>(string_value));
	} else if (test_binary()) {
		auto binary_value = std::shared_ptr<BinaryAttr>(new BinaryAttr(LT(1)));
		value = std::shared_ptr<AttrValue<BinaryAttr>>(new AttrValue<BinaryAttr>(binary_value));

		match({Token::category::true_, Token::category::false_});
	} else {
		auto simple_value = std::shared_ptr<SimpleAttr>(new SimpleAttr(LT(1)));
		value = std::shared_ptr<AttrValue<SimpleAttr>>(new AttrValue<SimpleAttr>(simple_value));

		if (LA(1) == Token::category::number) {
			match(Token::category::number);
		} else if (LA(1) == Token::category::size) {
			match(Token::category::size);
		} else {
			throw std::runtime_error(std::string(LT(1).begin()) + ": Unknown attr type: " + LT(1).value());
		}
	}

	return std::shared_ptr<Attr>(new Attr(name, id, value));
}

std::shared_ptr<AttrBlock> Parser::attr_block() {
	Token lbrace = LT(1);

	match({Token::category::lbrace, Token::category::lbracket});

	newline_list();
	std::vector<std::shared_ptr<Attr>> attrs;

	while (LA(1) == Token::category::id) {
		attrs.push_back(attr());
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

std::shared_ptr<AST::Stmt<AST::Controller>> Parser::controller() {
	Token controller = LT(1);

	match ({Token::category::machine, Token::category::flash, Token::category::network});

	Token name = LT(1);
	match(Token::category::id);

	newline_list();
	if (LA(1) != Token::category::lbrace) {
		throw std::runtime_error(std::string(LT(1).begin()) + ":Error: expected attribute block");
	}
	auto block = attr_block();
	auto stmt = std::shared_ptr<AST::Controller>(new AST::Controller(controller, name, block));

	return std::shared_ptr<AST::Stmt<AST::Controller>>(new AST::Stmt<AST::Controller>(stmt));
}

std::shared_ptr<Cmd> Parser::command() {
	Token vm = LT(1);
	match(Token::category::id);

	std::shared_ptr<IAction> act = action();
	return std::shared_ptr<Cmd>(new Cmd(vm, act));
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
	Token times = Token();
	auto combination = key_combination();

	if (LA(1) == Token::category::asterisk) {
		match(Token::category::asterisk);
		times = LT(1);
		match(Token::category::number);
	}

	return std::shared_ptr<KeySpec>(new KeySpec(combination, times));
}

std::shared_ptr<IAction> Parser::action() {
	std::shared_ptr<IAction> action;
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
	} else if (LA(1) == Token::category::id) {
		action = macro_call();
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
		action->set_delim(delim);
	}

	return action;
}

std::shared_ptr<Action<Empty>> Parser::empty_action() {
	match({Token::category::semi, Token::category::newline});
	auto action = std::shared_ptr<Empty>(new Empty());
	return std::shared_ptr<Action<Empty>>(new Action<Empty>(action));
}

std::shared_ptr<Action<Abort>> Parser::abort() {
	Token abort_token = LT(1);
	match(Token::category::abort);

	Token value = LT(1);

	auto message = string();

	auto action = std::shared_ptr<Abort>(new Abort(abort_token, message));
	return std::shared_ptr<Action<Abort>>(new Action<Abort>(action));
}

std::shared_ptr<Action<Print>> Parser::print() {
	Token print_token = LT(1);
	match(Token::category::print);

	Token value = LT(1);

	auto message = string();

	auto action = std::shared_ptr<Print>(new Print(print_token, message));
	return std::shared_ptr<Action<Print>>(new Action<Print>(action));
}

std::shared_ptr<Action<Type>> Parser::type() {
	Token type_token = LT(1);
	match(Token::category::type_);

	Token value = LT(1);
	auto text = string();

	std::shared_ptr<StringTokenUnion> interval = nullptr;

	if (test_string() || LA(1) == Token::category::interval) {
		match(Token::category::interval);
		interval = string_token_union(Token::category::time_interval);
	}
	auto action = std::shared_ptr<Type>(new Type(type_token, text, interval));
	return std::shared_ptr<Action<Type>>(new Action<Type>(action));
}

std::shared_ptr<Action<Wait>> Parser::wait() {
	Token wait_token = LT(1);
	match(Token::category::wait);

	std::shared_ptr<ISelectExpr> select_expression(nullptr);
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

	auto action = std::shared_ptr<Wait>(new Wait(wait_token, select_expression, timeout, interval));
	return std::shared_ptr<Action<Wait>>(new Action<Wait>(action));
}

std::shared_ptr<Action<Sleep>> Parser::sleep() {
	Token sleep_token = LT(1);
	match(Token::category::sleep);

	auto timeout = string_token_union(Token::category::time_interval);

	auto action = std::shared_ptr<Sleep>(new Sleep(sleep_token, timeout));
	return std::shared_ptr<Action<Sleep>>(new Action<Sleep>(action));
}

std::shared_ptr<Action<Press>> Parser::press() {
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

	auto action = std::shared_ptr<Press>(new Press(press_token, keys, interval));
	return std::shared_ptr<Action<Press>>(new Action<Press>(action));
}

std::shared_ptr<Action<Hold>> Parser::hold() {
	Token hold_token = LT(1);
	match(Token::category::hold);

	auto combination = key_combination();

	auto action = std::shared_ptr<Hold>(new Hold(hold_token, combination));
	return std::shared_ptr<Action<Hold>>(new Action<Hold>(action));
}

std::shared_ptr<Action<Release>> Parser::release() {
	Token release_token = LT(1);
	match(Token::category::release);

	std::shared_ptr<AST::KeyCombination> combination = nullptr;
	if (LA(1) == Token::category::id) {
		combination = key_combination();
	}

	auto action = std::shared_ptr<Release>(new Release(release_token, combination));
	return std::shared_ptr<Action<Release>>(new Action<Release>(action));
}

std::shared_ptr<AST::Action<AST::Mouse>> Parser::mouse() {
	Token mouse_token = LT(1);
	match(Token::category::mouse);

	std::shared_ptr<IMouseEvent> event = nullptr;

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

	auto action = std::make_shared<Mouse>(mouse_token, event);
	return std::make_shared<Action<Mouse>>(action);
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

std::shared_ptr<MouseMoveTarget<MouseSelectable>> Parser::mouse_selectable() {
	auto select = selectable();

	std::vector<std::shared_ptr<MouseAdditionalSpecifier>> specifiers;

	auto select_end = select->end();
	auto tmp = LT(1);

	for (Pos it = select->end(); LA(1) == Token::category::dot && Pos::is_adjacent(it, LT(1).begin());) {
		auto specifier = mouse_additional_specifier();
		specifiers.push_back(specifier);
		it = specifier->end();
	}

	Token timeout;

	if (LA(1) == Token::category::timeout) {
		match(Token::category::timeout);
		timeout = LT(1);
		match(Token::category::time_interval);
	}

	auto mouse_selectable = std::make_shared<MouseSelectable>(select, specifiers, timeout);
	return std::make_shared<MouseMoveTarget<MouseSelectable>>(mouse_selectable);
}

std::shared_ptr<AST::MouseEvent<AST::MouseMoveClick>> Parser::mouse_move_click() {
	Token event_token = LT(1);
	match({Token::category::click,
		Token::category::lclick,
		Token::category::move,
		Token::category::rclick,
		Token::category::mclick,
		Token::category::dclick});

	std::shared_ptr<IMouseMoveTarget> target = nullptr;

	if (test_selectable()) {
		target = mouse_selectable();
	} else if (LA(1) == Token::category::number) {
		target = mouse_coordinates();
	}

	if (event_token.type() == Token::category::move && !target) {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: you must specify a target to move the mouse cursor");
	}

	auto move_click = std::make_shared<MouseMoveClick>(event_token, target);
	return std::make_shared<MouseEvent<MouseMoveClick>>(move_click);
}

std::shared_ptr<AST::MouseEvent<AST::MouseHold>> Parser::mouse_hold() {
	Token event_token = LT(1);
	match(Token::category::hold);

	Token button = LT(1);
	match({Token::category::lbtn, Token::category::rbtn, Token::category::mbtn});

	auto move_hold = std::make_shared<MouseHold>(event_token, button);
	return std::make_shared<MouseEvent<MouseHold>>(move_hold);
}

std::shared_ptr<AST::MouseEvent<AST::MouseRelease>> Parser::mouse_release() {
	Token event_token = LT(1);
	match(Token::category::release);

	auto move_release = std::make_shared<MouseRelease>(event_token);
	return std::make_shared<MouseEvent<MouseRelease>>(move_release);
}

std::shared_ptr<AST::MouseEvent<AST::MouseWheel>> Parser::mouse_wheel() {
	Token event_token = LT(1);
	match(Token::category::wheel);

	Token direction = LT(1);

	if (direction.value() != "up" && direction.value() != "down") {
		throw std::runtime_error(std::string(direction.begin()) + " : Error: unknown wheel direction: " + direction.value());
	}

	match(Token::category::id);

	auto mouse_wheel = std::make_shared<MouseWheel>(event_token, direction);
	return std::make_shared<MouseEvent<MouseWheel>>(mouse_wheel);
}

std::shared_ptr<MouseMoveTarget<MouseCoordinates>> Parser::mouse_coordinates() {
	auto dx = LT(1);
	match(Token::category::number);
	auto dy = LT(1);
	match(Token::category::number);

	auto target = std::make_shared<MouseCoordinates>(dx, dy);
	return std::make_shared<MouseMoveTarget<MouseCoordinates>>(target);
}

std::shared_ptr<Action<Plug>> Parser::plug() {
	Token plug_token = LT(1);

	if (LA(1) == Token::category::plug) {
		match(Token::category::plug);
	} else {
		match(Token::category::unplug);
	}

	Token type = LT(1);
	if (LA(1) == Token::category::flash) {
		match(Token::category::flash);
	} else if (LA(1) == Token::category::dvd) {
		match(Token::category::dvd);
	}
	else {
		if (LT(1).value() != "nic" && LT(1).value() != "link") {
			throw std::runtime_error(std::string(LT(1).begin()) + ": Error: Unknown device type for plug/unplug");
		}
		match(Token::category::id);
	}

	Token name = Token();

	std::shared_ptr<String> path(nullptr);

	if (type.type() == Token::category::dvd) {
		if (plug_token.type() == Token::category::plug) {
			path = string();
		} //else this should be the end of unplug commands
	} else {
		name = LT(1);
		match(Token::category::id);
	}

	auto action = std::shared_ptr<Plug>(new Plug(plug_token, type, name, path));
	return std::shared_ptr<Action<Plug>>(new Action<Plug>(action));
}

std::shared_ptr<Action<Start>> Parser::start() {
	Token start_token = LT(1);
	match(Token::category::start);

	auto action = std::shared_ptr<Start>(new Start(start_token));
	return std::shared_ptr<Action<Start>>(new Action<Start>(action));
}

std::shared_ptr<Action<Stop>> Parser::stop() {
	Token stop_token = LT(1);
	match(Token::category::stop);

	auto action = std::shared_ptr<Stop>(new Stop(stop_token));
	return std::shared_ptr<Action<Stop>>(new Action<Stop>(action));
}

std::shared_ptr<Action<Shutdown>> Parser::shutdown() {
	Token shutdown_token = LT(1);
	match(Token::category::shutdown);

	std::shared_ptr<StringTokenUnion> timeout = nullptr;

	if (LA(1) == Token::category::timeout) {
		match(Token::category::timeout);
		timeout = string_token_union(Token::category::time_interval);
	}

	auto action = std::shared_ptr<Shutdown>(new Shutdown(shutdown_token, timeout));
	return std::shared_ptr<Action<Shutdown>>(new Action<Shutdown>(action));
}

std::shared_ptr<Action<Exec>> Parser::exec() {
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

	auto action = std::shared_ptr<Exec>(new Exec(exec_token, process_token, commands, timeout));
	return std::shared_ptr<Action<Exec>>(new Action<Exec>(action));
}

std::shared_ptr<Action<Copy>> Parser::copy() {
	Token copy_token = LT(1);
	match({Token::category::copyto, Token::category::copyfrom});

	auto from = string();
	auto to = string();

	std::shared_ptr<StringTokenUnion> timeout = nullptr;

	if (LA(1) == Token::category::timeout) {
		match(Token::category::timeout);
		timeout = string_token_union(Token::category::time_interval);
	}

	auto action = std::shared_ptr<Copy>(new Copy(copy_token, from, to, timeout));
	return std::shared_ptr<Action<Copy>>(new Action<Copy>(action));
}

std::shared_ptr<Action<ActionBlock>> Parser::action_block() {
	Token lbrace = LT(1);
	match(Token::category::lbrace);

	newline_list();
	std::vector<std::shared_ptr<IAction>> actions;

	while (test_action()) {
		auto act = action();
		actions.push_back(act);
		newline_list();
	}

	Token rbrace = LT(1);
	match(Token::category::rbrace);

	auto action = std::shared_ptr<ActionBlock>(new ActionBlock(lbrace, rbrace,  actions));
	return std::shared_ptr<Action<ActionBlock>>(new Action<ActionBlock>(action));
}

std::shared_ptr<Action<MacroCall>> Parser::macro_call() {
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

	auto action = std::shared_ptr<MacroCall>(new MacroCall(macro_name, params));
	return std::shared_ptr<Action<MacroCall>>(new Action<MacroCall>(action));
}

std::shared_ptr<Action<IfClause>> Parser::if_clause() {
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
	std::shared_ptr<IAction> else_action = nullptr;

	if (LA(1) == Token::category::else_) {
		else_token = LT(1);
		match(Token::category::else_);
		newline_list();
		else_action = action();
	}

	auto action = std::shared_ptr<IfClause>(new IfClause(
		if_token,
		open_paren, expression,
		close_paren, if_action,
		else_token, else_action
	));

	return std::shared_ptr<Action<IfClause>>(new Action<IfClause>(action));
}

std::shared_ptr<CounterList<Range>> Parser::range() {
	Token range_token = LT(1);
	match(Token::category::RANGE);

	std::shared_ptr<String> r1 = string();
	std::shared_ptr<String> r2 = nullptr;
	if (test_string()) {
		r2 = string();
	}

	auto counter_list = std::shared_ptr<Range>(new Range(range_token, r1, r2));
	return std::shared_ptr<CounterList<Range>>(new CounterList<Range>(counter_list));
}

std::shared_ptr<ICounterList> Parser::counter_list() {
	if (LA(1) == Token::category::RANGE) {
		return range();
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: Unknown counter_list specifier: " + LT(1).value());
	}
}

std::shared_ptr<Action<ForClause>> Parser::for_clause() {
	Token for_token = LT(1);
	match(Token::category::for_);

	match(Token::category::lparen);
	Token counter = LT(1);

	match(Token::category::id);

	match(Token::category::IN_);

	if (!test_counter_list()) {
		throw std::runtime_error(std::string(LT(1).begin()) + " : Error: expexted a RANGE");
	}

	std::shared_ptr<ICounterList> list = counter_list();
	match(Token::category::rparen);
	newline_list();

	auto cycle_body = action();

	Token else_token = Token();
	std::shared_ptr<IAction> else_action = nullptr;

	if (LA(1) == Token::category::else_) {
		else_token = LT(1);
		match(Token::category::else_);
		newline_list();
		else_action = action();
	}

	auto action = std::shared_ptr<ForClause>(new ForClause(
		for_token,
		counter,
		list,
		cycle_body,
		else_token,
		else_action
	));
	return std::shared_ptr<Action<ForClause>>(new Action<ForClause>(action));
}

std::shared_ptr<Action<CycleControl>> Parser::cycle_control() {
	Token control_token = LT(1);
	match({Token::category::break_, Token::category::continue_});

	auto action = std::shared_ptr<CycleControl>(new CycleControl(control_token));
	return std::shared_ptr<Action<CycleControl>>(new Action<CycleControl>(action));
}

std::shared_ptr<ISelectExpr> Parser::select_expr() {
	auto left = std::shared_ptr<SelectExpr<ISelectable>>(new SelectExpr<ISelectable>(selectable()));
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

std::shared_ptr<AST::SelectExpr<AST::SelectBinOp>> Parser::select_binop(std::shared_ptr<AST::ISelectExpr> left) {
	auto op = LT(1);

	match({Token::category::double_ampersand, Token::category::double_vertical_bar});
	newline_list();

	auto right = select_expr();

	auto binop = std::shared_ptr<AST::SelectBinOp>(new AST::SelectBinOp(left, op, right));
	return std::shared_ptr<AST::SelectExpr<AST::SelectBinOp>>(new AST::SelectExpr<AST::SelectBinOp>(binop));
}

std::shared_ptr<ISelectable> Parser::selectable() {
	auto not_token = Token();
	if (LA(1) == Token::category::exclamation_mark) {
		not_token = LT(1);
		match(Token::category::exclamation_mark);
	}

	if (test_string()) {
		return std::shared_ptr<Selectable<SelectText>>(new Selectable<SelectText>(not_token, select_text()));
	} else if(LA(1) == Token::category::js) {
		return std::shared_ptr<Selectable<SelectJS>>(new Selectable<SelectJS>(not_token, select_js()));
	} else if(LA(1) == Token::category::lparen) {
		return std::shared_ptr<Selectable<SelectParentedExpr>>(new Selectable<SelectParentedExpr>(not_token, select_parented_expr()));
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + ":Error: Unknown selective object type: " + LT(1).value());
	}
}

std::shared_ptr<SelectJS> Parser::select_js() {
	Token js = LT(1);
	match(Token::category::js);
	auto script = string();
	return std::shared_ptr<SelectJS>(new SelectJS(js, script));
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

	auto new_node = std::shared_ptr<String>(new String(str));

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
		throw std::runtime_error(std::string(LT(1)) + ": Error: expected a string or " + Token::type_to_string(expected_token_type) + ", but got " +
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

std::shared_ptr<IFactor> Parser::factor() {
	auto not_token = Token();
	if (LA(1) == Token::category::NOT) {
		not_token = LT(1);
		match(Token::category::NOT);
	}

	//TODO: newline
	if (LA(1) == Token::category::check) {
		return std::shared_ptr<Factor<Check>>(new Factor<Check>(not_token, check()));
	} else if(test_comparison()) {
		return std::shared_ptr<Factor<Comparison>>(new Factor<Comparison>(not_token, comparison()));
	} else if(test_defined()) {
		return std::shared_ptr<Factor<Defined>>(new Factor<Defined>(not_token, defined()));
	} else if (LA(1) == Token::category::lparen) {
		return std::shared_ptr<Factor<ParentedExpr>>(new Factor<ParentedExpr>(not_token, parented_expr()));
	} else if (test_string()) {
		return std::shared_ptr<Factor<String>>(new Factor<String>(not_token, string()));
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: Unknown expression: " + LT(1).value());
	}
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

	std::shared_ptr<ISelectExpr> select_expression(nullptr);

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

	return std::shared_ptr<Check>(new Check(check_token, select_expression, timeout, interval));
}

std::shared_ptr<Expr<BinOp>> Parser::binop(std::shared_ptr<IExpr> left) {
	auto op = LT(1);

	match({Token::category::OR, Token::category::AND});
	newline_list();

	auto right = expr();

	auto binop = std::shared_ptr<BinOp>(new BinOp(op, left, right));
	return std::shared_ptr<Expr<BinOp>>(new Expr<BinOp>(binop));
}

std::shared_ptr<IExpr> Parser::expr() {
	auto left = std::shared_ptr<Expr<IFactor>>(new Expr<IFactor>(factor()));

	if ((LA(1) == Token::category::AND) ||
		(LA(1) == Token::category::OR)) {
		return binop(left);
	} else {
		return left;
	}
}

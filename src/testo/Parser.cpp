
#include "Parser.hpp"
#include "Utils.hpp"
#include "TemplateParser.hpp"
#include <fstream>

using namespace AST;

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
		throw std::runtime_error(std::string(LT(1).pos()) +
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

	throw std::runtime_error(std::string(LT(1).pos()) +
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
		(LA(1) == Token::category::press) ||
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

bool Parser::test_string() const {
	return ((LA(1) == Token::category::quoted_string) ||
		(LA(1) == Token::category::triple_quoted_string));
}

bool Parser::test_selectable() const {
	return (test_string() || (LA(1) == Token::category::js));
}

bool Parser::test_select_expr() const {
	return (test_selectable() ||
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
			throw std::runtime_error(std::string(include_token.pos()) + ": fatal error: no such file: " + dest_file.generic_string());
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
			throw std::runtime_error(std::string(LT(1).pos()) + ":error: expected declaration or include");
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
		throw std::runtime_error(std::string(LT(1).pos())
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

std::shared_ptr<Stmt<Macro>> Parser::macro() {
	Token macro = LT(1);
	match(Token::category::macro);

	Token name = LT(1);
	match(Token::category::id);

	match(Token::category::lparen);

	std::vector<Token> params;

	if (LA(1) == Token::category::id) {
		params.push_back(LT(1));
		match(Token::category::id);
	}

	while (LA(1) == Token::category::comma) {
		if (params.empty()) {
			match(Token::category::rparen); //will cause failure
		}
		match(Token::category::comma);
		params.push_back(LT(1));
		match(Token::category::id);
	}

	match(Token::category::rparen);

	newline_list();
	auto actions = action_block();

	auto stmt = std::shared_ptr<Macro>(new Macro(macro, name, params, actions));
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
			throw std::runtime_error(std::string(LT(1).pos()) + ": Unknown attr type: " + LT(1).value());
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
		throw std::runtime_error(std::string(LT(1).pos()) + ":Error: expected attribute block");
	}
	auto block = attr_block();
	auto stmt = std::shared_ptr<AST::Controller>(new AST::Controller(controller, name, block));
	return std::shared_ptr<AST::Stmt<AST::Controller>>(new AST::Stmt<AST::Controller>(stmt));
}

std::shared_ptr<Cmd> Parser::command() {
	std::vector<Token> vms;
	vms.push_back(LT(1));
	match(Token::category::id);

	while (LA(1) == Token::category::comma) {
		match(Token::category::comma);
		vms.push_back(LT(1));
		match(Token::category::id);
	}

	newline_list();
	std::shared_ptr<IAction> act = action();
	return std::shared_ptr<Cmd>(new Cmd(vms, act));
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

std::shared_ptr<KeySpec> Parser::key_spec() {
	std::vector<Token> buttons;
	Token times = Token();

	do {
		buttons.push_back(LT(1));
		match(Token::category::id);

		if (LA(1) == Token::category::plus) {
			match(Token::category::plus);
		}
	} while (LA(1) == Token::category::id);

	if (LA(1) == Token::category::asterisk) {
		match(Token::category::asterisk);
		times = LT(1);
		match(Token::category::number);
	}

	return std::shared_ptr<KeySpec>(new KeySpec(buttons, times));
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
	} else if (LA(1) == Token::category::press) {
		action = press();
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
		throw std::runtime_error(std::string(LT(1).pos()) + ":Error: Unknown action: " + LT(1).value());
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
			throw std::runtime_error(std::string(LT(1).pos()) +
				": Expected new line or ';' \"");
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

	auto action = std::shared_ptr<Type>(new Type(type_token, text));
	return std::shared_ptr<Action<Type>>(new Action<Type>(action));
}

std::shared_ptr<Action<Wait>> Parser::wait() {
	Token wait_token = LT(1);
	match(Token::category::wait);

	std::shared_ptr<ISelectExpr> select_expression(nullptr);
	Token timeout = Token();
	Token time_interval = Token();

	if (test_select_expr()) {
		select_expression = select_expr();
	}

	//special check for multiline strings. We don't support them yet.

	//ToDo: Thiple check this part
	if (select_expression && (select_expression->t.type() == Token::category::triple_quoted_string)) {
		throw std::runtime_error(std::string(select_expression->begin()) +
			": Error: multiline strings are not supported in wait action");
	}

	if (LA(1) == Token::category::timeout) {
		timeout = LT(1);
		match(Token::category::timeout);

		time_interval = LT(1);
		match(Token::category::time_interval);
	}

	if (!(select_expression || timeout)) {
		throw std::runtime_error(std::string(wait_token.pos()) +
			": Error: either TEXT or FOR (of both) must be specified for wait command");
	}

	auto action = std::shared_ptr<Wait>(new Wait(wait_token, select_expression, timeout, time_interval));
	return std::shared_ptr<Action<Wait>>(new Action<Wait>(action));
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

	auto action = std::shared_ptr<Press>(new Press(press_token, keys));
	return std::shared_ptr<Action<Press>>(new Action<Press>(action));
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
	} else {
		throw std::runtime_error(std::string(LT(1).pos()) + " : Error: unknown mouse action: " + LT(1).value());
	}

	auto action = std::shared_ptr<Mouse>(new Mouse(mouse_token, event));
	return std::shared_ptr<Action<Mouse>>(new Action<Mouse>(action));
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
	bool is_coordinates = false;

	if (test_selectable()) {
		auto object = selectable();
		target = std::shared_ptr<MouseMoveTarget<ISelectable>>(new MouseMoveTarget<ISelectable>(object));
	} else if (LA(1) == Token::category::number) {
		is_coordinates = true;
		target = mouse_coordinates();
	}

	Token timeout = Token();

	if (LA(1) == Token::category::timeout) {
		match(Token::category::timeout);

		timeout = LT(1);
		match(Token::category::time_interval);
	}

	if (timeout && (target == nullptr)) {
		throw std::runtime_error(std::string(timeout.pos()) + ": Error: timeout can be used only with an object");
	}

	if (timeout && is_coordinates) {
		throw std::runtime_error(std::string(timeout.pos()) + ": Error: timeout can't be used with coordinates");
	}

	auto move_click = std::shared_ptr<MouseMoveClick>(new MouseMoveClick(event_token, target, timeout));
	return std::shared_ptr<MouseEvent<MouseMoveClick>>(new MouseEvent(move_click));
}

std::shared_ptr<AST::MouseEvent<AST::MouseHold>> Parser::mouse_hold() {
	Token event_token = LT(1);
	match(Token::category::hold);

	Token button = LT(1);
	match({Token::category::lbtn, Token::category::rbtn, Token::category::mbtn});

	auto move_hold = std::shared_ptr<MouseHold>(new MouseHold(event_token, button));
	return std::shared_ptr<MouseEvent<MouseHold>>(new MouseEvent(move_hold));
}

std::shared_ptr<AST::MouseEvent<AST::MouseRelease>> Parser::mouse_release() {
	Token event_token = LT(1);
	match(Token::category::release);

	auto move_release = std::shared_ptr<MouseRelease>(new MouseRelease(event_token));
	return std::shared_ptr<MouseEvent<MouseRelease>>(new MouseEvent(move_release));
}

std::shared_ptr<MouseMoveTarget<MouseCoordinates>> Parser::mouse_coordinates() {
	auto dx = LT(1);
	match(Token::category::number);
	auto dy = LT(1);
	match(Token::category::number);

	auto target = std::shared_ptr<MouseCoordinates>(new MouseCoordinates(dx, dy));
	return std::shared_ptr<MouseMoveTarget<MouseCoordinates>>(new MouseMoveTarget<MouseCoordinates>(target));
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
			throw std::runtime_error(std::string(LT(1).pos()) + ": Error: Unknown device type for plug/unplug");
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

	Token timeout = Token();
	Token time_interval = Token();

	if (LA(1) == Token::category::timeout) {
		timeout = LT(1);
		match(Token::category::timeout);

		time_interval = LT(1);
		match(Token::category::time_interval);
	}

	auto action = std::shared_ptr<Shutdown>(new Shutdown(shutdown_token, timeout, time_interval));
	return std::shared_ptr<Action<Shutdown>>(new Action<Shutdown>(action));
}

std::shared_ptr<Action<Exec>> Parser::exec() {
	Token exec_token = LT(1);
	match(Token::category::exec);

	Token process_token = LT(1);
	match(Token::category::id);

	auto commands = string();

	Token timeout = Token();
	Token time_interval = Token();

	if (LA(1) == Token::category::timeout) {
		timeout = LT(1);
		match(Token::category::timeout);

		time_interval = LT(1);
		match(Token::category::time_interval);
	}

	auto action = std::shared_ptr<Exec>(new Exec(exec_token, process_token, commands, timeout, time_interval));
	return std::shared_ptr<Action<Exec>>(new Action<Exec>(action));
}

std::shared_ptr<Action<Copy>> Parser::copy() {
	Token copy_token = LT(1);
	match({Token::category::copyto, Token::category::copyfrom});

	auto from = string();
	auto to = string();

	Token timeout = Token();
	Token time_interval = Token();

	if (LA(1) == Token::category::timeout) {
		timeout = LT(1);
		match(Token::category::timeout);

		time_interval = LT(1);
		match(Token::category::time_interval);
	}

	auto action = std::shared_ptr<Copy>(new Copy(copy_token, from, to, timeout, time_interval));
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

std::shared_ptr<Action<ForClause>> Parser::for_clause() {
	Token for_token = LT(1);
	match(Token::category::for_);

	Token counter = LT(1);
	match(Token::category::id);

	Token in = LT(1);
	match(Token::category::in);

	Token begin = LT(1);
	match(Token::category::number);

	Token double_dot = LT(1);
	match(Token::category::double_dot);

	Token end = LT(1);
	match(Token::category::number);

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
		in,
		begin,
		double_dot,
		end,
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
	if (LA(1) == Token::category::exclamation_mark) {
		return select_unop();
	}

	if (LA(1) == Token::category::lparen) {
		return select_parented_expr();
	}

	std::shared_ptr<ISelectExpr> left = std::shared_ptr<SelectExpr<ISelectable>>(new SelectExpr<ISelectable>(selectable()));

	if ((LA(1) == Token::category::double_ampersand) ||
		(LA(1) == Token::category::double_vertical_bar)) {
		return select_binop(left);
	} else {
		return left;
	}
}


std::shared_ptr<AST::SelectExpr<AST::SelectUnOp>> Parser::select_unop() {
	auto op = LT(1);

	match(Token::category::exclamation_mark);

	auto expression = select_expr();

	auto unop = std::shared_ptr<AST::SelectUnOp>(new AST::SelectUnOp(op, expression));
	return std::shared_ptr<AST::SelectExpr<AST::SelectUnOp>>(new AST::SelectExpr<AST::SelectUnOp>(unop));
}

std::shared_ptr<AST::SelectExpr<AST::SelectParentedExpr>> Parser::select_parented_expr() {
	auto lparen = LT(1);
	match(Token::category::lparen);

	auto expression = select_expr();

	auto rparen = LT(1);
	match(Token::category::rparen);
	auto parented_expr = std::shared_ptr<AST::SelectParentedExpr>(new AST::SelectParentedExpr(lparen, expression, rparen));
	return std::shared_ptr<AST::SelectExpr<AST::SelectParentedExpr>>(new AST::SelectExpr<AST::SelectParentedExpr>(parented_expr));
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
	std::shared_ptr<ISelectable> query;
	if (test_string()) {
		query = std::shared_ptr<Selectable<String>>(new Selectable<String>(string()));
	} else if(LA(1) == Token::category::js) {
		query = select_js();
	} else {
		throw std::runtime_error(std::string(LT(1).pos()) + ":Error: Unknown selective object type: " + LT(1).value());
	}

	return query;
}

std::shared_ptr<Selectable<SelectJS>> Parser::select_js() {
	Token js = LT(1);
	match(Token::category::js);
	auto script = string();
	auto select_js = std::shared_ptr<SelectJS>(new SelectJS(js, script));

	return std::shared_ptr<Selectable<SelectJS>>(new Selectable<SelectJS>(select_js));
}

std::shared_ptr<String> Parser::string() {
	Token str = LT(1);
	if (!test_string()) {
		throw std::runtime_error(std::string(LT(1).pos()) + ": Error: expected string");
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
	} else if (LA(1) == Token::category::lparen) {
		match(Token::category::lparen);
		auto result = std::shared_ptr<Factor<IExpr>>(new Factor<IExpr>(not_token, expr()));
		match(Token::category::rparen);
		return result;
	} else if (test_string()) {
		return std::shared_ptr<Factor<String>>(new Factor<String>(not_token, string()));
	} else {
		throw std::runtime_error(std::string(LT(1).pos()) + ":Error: Unknown expression: " + LT(1).value());
	}
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

std::shared_ptr<Check> Parser::check() {
	Token check_token = LT(1);
	match(Token::category::check);

	std::shared_ptr<ISelectExpr> select_expression(nullptr);
	select_expression = select_expr();

	Token timeout, time_interval;

	if (LA(1) == Token::category::timeout) {
		timeout = LT(1);
		match(Token::category::timeout);

		time_interval = LT(1);
		match(Token::category::time_interval);
	}

	return std::shared_ptr<Check>(new Check(check_token, select_expression, timeout, time_interval));
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


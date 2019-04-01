
#include "Parser.hpp"
#include "Utils.hpp"

using namespace AST;

Parser::Parser(const fs::path& file)
{
	Ctx ctx(file);
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

bool Parser::test_assignment() const {
	return ((LA(1) == Token::category::id) &&
		LA(2) == Token::category::assign);
}

bool Parser::test_stmt() const {
	return ((LA(1) == Token::category::snapshot) ||
		(LA(1) == Token::category::test) ||
		(LA(1) == Token::category::macro) ||
		test_controller());
}

bool Parser::test_include() const {
	return (LA(1) == Token::category::include);
}

bool Parser::test_controller() const {
	return (LA(1) == Token::category::machine) ||
		(LA(1) == Token::category::flash);
}

bool Parser::test_command() const {
	return (LA(1) == Token::category::id);
}

bool Parser::test_action() const {
	return ((LA(1) == Token::category::type_) ||
		(LA(1) == Token::category::wait) ||
		(LA(1) == Token::category::press) ||
		(LA(1) == Token::category::plug) ||
		(LA(1) == Token::category::unplug) ||
		(LA(1) == Token::category::start) ||
		(LA(1) == Token::category::stop) ||
		(LA(1) == Token::category::exec) ||
		(LA(1) == Token::category::set) ||
		(LA(1) == Token::category::copyto) ||
		(LA(1) == Token::category::lbrace) ||
		(LA(1) == Token::category::if_) ||
		(LA(1) == Token::category::for_) ||
		(LA(1) == Token::category::break_) ||
		(LA(1) == Token::category::continue_) ||
		(LA(1) == Token::category::semi) ||
		(LA(1) == Token::category::id)); //macro call
}

bool Parser::test_word() const {
	return ((LA(1) == Token::category::dbl_quoted_string) ||
		(LA(1) == Token::category::var_ref) ||
		(LA(1) == Token::category::multiline_string));
}

bool Parser::test_comparison() const {
	if (test_word()) {
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
	match(Token::category::dbl_quoted_string);
	match(Token::category::newline);
	fs::path dest_file = dest_file_token.value().substr(1, dest_file_token.value().length() - 2);

	if (dest_file.is_relative()) {
		fs::path combined = lexers[lexers.size() - 1].lex.file().parent_path() / dest_file;
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

	Ctx new_ctx(dest_file);
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
	if (LA(1) == Token::category::snapshot) {
		return snapshot();
	} else if (LA(1) == Token::category::test) {
		return test();
	} else if (LA(1) == Token::category::macro) {
		return macro();
	} else if (test_controller()) {
		return controller();
	} else {
		throw std::runtime_error(std::string(LT(1).pos())
			+ ": Error: unsupported statement: " + LT(1).value());
	}
}

std::shared_ptr<Stmt<Snapshot>> Parser::snapshot() {
	Token snapshot = LT(1);
	match(Token::category::snapshot);

	Token name = LT(1);
	match(Token::category::id);
	Token parent = Token();

	if (LA(1) == Token::category::colon) {
		match(Token::category::colon);
		parent = LT(1);
		match(Token::category::id);
	}

	newline_list();
	auto actions = action_block();

	auto stmt = std::shared_ptr<Snapshot>(new Snapshot(snapshot, name, parent, actions));
	return std::shared_ptr<Stmt<Snapshot>>(new Stmt<Snapshot>(stmt));
}

std::shared_ptr<Stmt<Test>> Parser::test() {
	Token test = LT(1);
	match(Token::category::test);

	Token name = LT(1);
	match(Token::category::id);
	match(Token::category::colon);

	std::vector<std::shared_ptr<VmState>> vms;
	newline_list();
	vms.push_back(vm_state());

	while (LA(1) == Token::category::comma) {
		match(Token::category::comma);
		newline_list();
		vms.push_back(vm_state());
	}

	newline_list();
	auto commands = command_block();
	auto stmt = std::shared_ptr<Test>(new Test(test, name, vms, commands));
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

std::shared_ptr<VmState> Parser::vm_state() {
	Token name = LT(1);

	match (Token::category::id);

	Token snapshot = Token();

	if (LA(1) == Token::category::lparen) {
		match(Token::category::lparen);
		snapshot = LT(1);
		match(Token::category::id);
		match(Token::category::rparen);
	}

	return std::shared_ptr<VmState>(new VmState(name, snapshot));
}

std::shared_ptr<Assignment> Parser::assignment() {
	Token left = LT(1);
	match(Token::category::id);

	Token assign = LT(1);
	match(Token::category::assign);

	auto right = word();
	return std::shared_ptr<Assignment>(new Assignment(left, assign, right));
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
	} else if (test_word()) {
		auto word_value = std::shared_ptr<WordAttr>(new WordAttr(word()));
		value = std::shared_ptr<AttrValue<WordAttr>>(new AttrValue<WordAttr>(word_value));
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
	match(Token::category::lbrace);

	newline_list();
	std::vector<std::shared_ptr<Attr>> attrs;

	while (LA(1) == Token::category::id) {
		attrs.push_back(attr());
		match(Token::category::newline);
		newline_list();
	}

	newline_list();
	Token rbrace = LT(1);
	match(Token::category::rbrace);

	return std::shared_ptr<AttrBlock>(new AttrBlock(lbrace, rbrace, attrs));
}

std::shared_ptr<Stmt<Controller>> Parser::controller() {
	Token controller = LT(1);
	if (LA(1) == Token::category::machine) {
		match(Token::category::machine);
	} else {
		match(Token::category::flash);
	}

	Token name = LT(1);
	match(Token::category::id);

	newline_list();
	auto block = attr_block();
	auto stmt = std::shared_ptr<Controller>(new Controller(controller, name, block));
	return std::shared_ptr<Stmt<Controller>>(new Stmt<Controller>(stmt));
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
	if (LA(1) == Token::category::type_) {
		action = type();
	} else if (LA(1) == Token::category::wait) {
		action = wait();
	} else if (LA(1) == Token::category::press) {
		action = press();
	} else if ((LA(1) == Token::category::plug) || (LA(1) == Token::category::unplug)) {
		action = plug();
	} else if (LA(1) == Token::category::start) {
		action = start();
	} else if (LA(1) == Token::category::stop) {
		action = stop();
	} else if (LA(1) == Token::category::exec) {
		action = exec();
	} else if (LA(1) == Token::category::set) {
		action = set();
	} else if (LA(1) == Token::category::copyto) {
		action = copyto();
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

std::shared_ptr<Action<Type>> Parser::type() {
	Token type_token = LT(1);
	match(Token::category::type_);

	Token value = LT(1);

	auto text = word();

	auto action = std::shared_ptr<Type>(new Type(type_token, text));
	return std::shared_ptr<Action<Type>>(new Action<Type>(action));
}

std::shared_ptr<Action<Wait>> Parser::wait() {
	Token wait_token = LT(1);
	match(Token::category::wait);

	std::shared_ptr<Word> value(nullptr);
	Token for_ = Token();
	Token time_interval = Token();

	if (test_word()) {
		value = word();
	}

	if (LA(1) == Token::category::for_) {
		for_ = LT(1);
		match(Token::category::for_);

		time_interval = LT(1);
		match(Token::category::time_interval);
	}

	if (!(value || for_)) {
		throw std::runtime_error(std::string(wait_token.pos()) +
			": Error: either TEXT or FOR (of both) must be specified for wait command");
	}

	auto action = std::shared_ptr<Wait>(new Wait(wait_token, value, for_, time_interval));
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

	std::shared_ptr<Word> path(nullptr);

	if (type.type() == Token::category::dvd) {
		if (plug_token.type() == Token::category::plug) {
			path = word();
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

std::shared_ptr<Action<Exec>> Parser::exec() {
	Token exec_token = LT(1);
	match(Token::category::exec);

	Token process_token = LT(1);
	match(Token::category::id);

	auto commands = word();

	auto action = std::shared_ptr<Exec>(new Exec(exec_token, process_token, commands));
	return std::shared_ptr<Action<Exec>>(new Action<Exec>(action));
}

std::shared_ptr<Action<Set>> Parser::set() {
	Token set_token = LT(1);
	match(Token::category::set);

	std::vector<std::shared_ptr<Assignment>> assignments;

	assignments.push_back(assignment());
	while (LA(1) == Token::category::comma) {
		match(Token::category::comma);
		newline_list();
		assignments.push_back(assignment());
	}

	auto action = std::shared_ptr<Set>(new Set(set_token, assignments));
	return std::shared_ptr<Action<Set>>(new Action<Set>(action));
}

std::shared_ptr<Action<CopyTo>> Parser::copyto() {
	Token copyto_token = LT(1);
	match(Token::category::copyto);

	auto from = word();
	auto to = word();

	auto action = std::shared_ptr<CopyTo>(new CopyTo(copyto_token, from, to));
	return std::shared_ptr<Action<CopyTo>>(new Action<CopyTo>(action));
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

	std::vector<std::shared_ptr<Word>> params;

	if (test_word()) {
		params.push_back(word());
	}

	while (LA(1) == Token::category::comma) {
		if (params.empty()) {
			match(Token::category::rparen); //will cause failure
		}
		match(Token::category::comma);
		params.push_back(word());
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
	auto action = std::shared_ptr<ForClause>(new ForClause(
		for_token,
		counter,
		in,
		begin,
		double_dot,
		end,
		cycle_body
	));
	return std::shared_ptr<Action<ForClause>>(new Action<ForClause>(action));
}

std::shared_ptr<Action<CycleControl>> Parser::cycle_control() {
	Token control_token = LT(1);
	match({Token::category::break_, Token::category::continue_});

	auto action = std::shared_ptr<CycleControl>(new CycleControl(control_token));
	return std::shared_ptr<Action<CycleControl>>(new Action<CycleControl>(action));
}

std::shared_ptr<Word> Parser::word() {
	std::vector<Token> parts;

	if (!test_word()) {
		throw std::runtime_error(std::string(LT(1).pos()) + ": Error: expected word specificator");
	}

	parts.push_back(LT(1));
	match({Token::category::dbl_quoted_string, Token::category::multiline_string, Token::category::var_ref});

	while (LA(1) == Token::category::plus) {
		match(Token::category::plus);
		newline_list();
		parts.push_back(LT(1));
		match({Token::category::dbl_quoted_string, Token::category::multiline_string, Token::category::var_ref});
	}

	return std::shared_ptr<Word>(new Word(parts));
}

std::shared_ptr<IFactor> Parser::factor() {
	auto not_token = Token();
	if (LA(1) == Token::category::NOT) {
		not_token = LT(1);
		match(Token::category::NOT);
	}

	//TODO: newline
	if(test_comparison()) {
		return std::shared_ptr<Factor<Comparison>>(new Factor<Comparison>(not_token, comparison()));
	} else if (LA(1) == Token::category::lparen) {
		match(Token::category::lparen);
		auto result = std::shared_ptr<Factor<IExpr>>(new Factor<IExpr>(not_token, expr()));
		match(Token::category::rparen);
		return result;
	} else if (test_word()) {
		return std::shared_ptr<Factor<Word>>(new Factor<Word>(not_token, word()));
	} else {
		throw std::runtime_error(std::string(LT(1).pos()) + ":Error: Unknown expression: " + LT(1).value());
	}
}

std::shared_ptr<Comparison> Parser::comparison() {
	auto left = word();

	Token op = LT(1);

	match({
		Token::category::GREATER,
		Token::category::LESS,
		Token::category::EQUAL,
		Token::category::STRGREATER,
		Token::category::STRLESS,
		Token::category::STREQUAL
		});

	auto right = word();

	return std::shared_ptr<Comparison>(new Comparison(op, left, right));
}

std::shared_ptr<Expr<BinOp>> Parser::binop(std::shared_ptr<IExpr> left) {
	auto op = LT(1);

	match({Token::category::OR, Token::category::AND});

	auto right = expr();

	auto binop = std::shared_ptr<BinOp>(new BinOp(op, left, right));
	return std::shared_ptr<Expr<BinOp>>(new Expr(binop));
}

std::shared_ptr<IExpr> Parser::expr() {
	auto left = std::shared_ptr<Expr<IFactor>>(new Expr(factor()));

	if ((LA(1) == Token::category::AND) ||
		(LA(1) == Token::category::OR)) {
		return binop(left);
	} else {
		return left;
	}
}


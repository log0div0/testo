
#include <Parser.hpp>
#include <Utils.hpp>

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
		(LA(1) == Token::category::lbrace));
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
			throw std::runtime_error(std::string(include_token.pos()) + ": fatal error: no such file: " + std::string(dest_file));
		}
		dest_file = fs::canonical(combined);
	}

	//check for cycles

	for (auto& ctx: lexers) {
		if (ctx.lex.file() == dest_file) {
			throw std::runtime_error(std::string(include_token.pos()) + ": fatal error: cyclic include detected: $include " + std::string(dest_file_token));
		}
	}

	Ctx new_ctx(dest_file);
	lexers.push_back(new_ctx);

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

	Token right = LT(1);
	match(Token::category::dbl_quoted_string);

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
	} else {
		auto simple_value = std::shared_ptr<SimpleAttr>(new SimpleAttr(LT(1)));
		value = std::shared_ptr<AttrValue<SimpleAttr>>(new AttrValue<SimpleAttr>(simple_value));

		if (LA(1) == Token::category::dbl_quoted_string) {
			match(Token::category::dbl_quoted_string);
		} else if (LA(1) == Token::category::number) {
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
	Token vm = LT(1);
	match(Token::category::id);

	newline_list();
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
	} else {
		throw std::runtime_error(std::string(LT(1).pos()) + ":Error: Unknown action: " + LT(1).value());
	}

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

	return action;
}

std::shared_ptr<Action<Type>> Parser::type() {
	Token type_token = LT(1);
	match(Token::category::type_);

	Token value = LT(1);
	match(Token::category::dbl_quoted_string);

	auto action = std::shared_ptr<Type>(new Type(type_token, value));
	return std::shared_ptr<Action<Type>>(new Action<Type>(action));
}

std::shared_ptr<Action<Wait>> Parser::wait() {
	Token wait_token = LT(1);
	match(Token::category::wait);

	Token value = Token();
	Token for_ = Token();
	Token time_interval = Token();

	if (LA(1) == Token::category::dbl_quoted_string) {
		value = LT(1);
		match(Token::category::dbl_quoted_string);
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
			throw std::runtime_error(std::string(LT(1).pos() + ": Error: Unknown device type for plug/unplug"));
		}
		match(Token::category::id);
	}

	Token name = LT(1);

	if (type.type() == Token::category::dvd) {
		if (plug_token.type() == Token::category::plug) {
			match(Token::category::dbl_quoted_string);
		} //else this should be the end of unplug commands
	} else {
		match(Token::category::id);
	}

	auto action = std::shared_ptr<Plug>(new Plug(plug_token, type, name));
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

	Token commands_token = LT(1);

	if (LA(1) == Token::category::dbl_quoted_string) {
		match(Token::category::dbl_quoted_string);
	} else if (LA(1) == Token::category::multiline_string) {
		match(Token::category::multiline_string);
	} else {
		throw std::runtime_error(std::string("unexpected token: ") + LT(1).value() + ", expected double qouted or multiline string");
	}

	auto action = std::shared_ptr<Exec>(new Exec(exec_token, process_token, commands_token));
	return std::shared_ptr<Action<Exec>>(new Action<Exec>(action));
}

std::shared_ptr<Action<Set>> Parser::set() {
	Token set_token = LT(1);
	match(Token::category::set);

	std::vector<std::shared_ptr<Assignment>> assignments;

	while (test_assignment()) {
		assignments.push_back(assignment());
	}

	if (!assignments.size()) {
		throw std::runtime_error(std::string(set_token.pos()) + ": Error: set action needs at least one assignment");
	}

	auto action = std::shared_ptr<Set>(new Set(set_token, assignments));
	return std::shared_ptr<Action<Set>>(new Action<Set>(action));
}

std::shared_ptr<Action<CopyTo>> Parser::copyto() {
	Token copyto_token = LT(1);
	match(Token::category::copyto);

	Token from = LT(1);
	match(Token::category::dbl_quoted_string);

	Token to = LT(1);
	match(Token::category::dbl_quoted_string);

	auto action = std::shared_ptr<CopyTo>(new CopyTo(copyto_token, from, to));
	return std::shared_ptr<Action<CopyTo>>(new Action<CopyTo>(action));
}

std::shared_ptr<Action<ActionBlock>> Parser::action_block() {
	Token lbrace = LT(1);
	match(Token::category::lbrace);

	newline_list();
	std::vector<std::shared_ptr<IAction>> actions;

	while (test_action()) {
		actions.push_back(action());
		newline_list();
	}

	Token rbrace = LT(1);
	match(Token::category::rbrace);

	auto action = std::shared_ptr<ActionBlock>(new ActionBlock(lbrace, rbrace,  actions));
	return std::shared_ptr<Action<ActionBlock>>(new Action<ActionBlock>(action));
}


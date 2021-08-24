
#include "Parser.hpp"
#include "Utils.hpp"
#include "Exceptions.hpp"
#include "TemplateLiterals.hpp"
#include <fstream>
#include <fmt/format.h>

using namespace AST;

struct UnknownOption: Exception {
	UnknownOption(const Token& name): Exception(std::string(name.begin()) + ": Error: Unknown option: " + name.value()) {}
};

struct UnknownAttr: Exception {
	UnknownAttr(const Token& name): Exception(std::string(name.begin()) + ": Error: Unknown attribute: " + name.value()) {}
};

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

Token Parser::eat(Token::category type) {
	return eat(std::vector<Token::category>{type});
}

Token Parser::eat(const std::vector<Token::category> types) {
	Token token = LT(1);
	for (auto type: types) {
		if (token.type() == type) {
			lexers.back().p++;
			return token;
		}
	}

	std::string error_msg = std::string(LT(1).begin()) +
		": Error: unexpected token \"" +
		LT(1).value() + "\", expected";

	if (types.size() == 1) {
		error_msg += ": " + Token::type_to_string(types[0]);
	} else {
		error_msg += " one of the following tokens: ";
		for (size_t i = 0; i != types.size(); ++i) {
			if (i) {
				error_msg += " or ";
			}
			error_msg += Token::type_to_string(types[i]);
		}
	}

	throw std::runtime_error(error_msg);
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
		eat(Token::category::newline);
	}
}

void Parser::handle_include() {
	//Get new Lexer
	auto include_token = eat(Token::category::include);

	auto dest_file_token = eat(Token::category::quoted_string);
	eat(Token::category::newline);
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

std::shared_ptr<AST::OptionSeq> Parser::option_seq(const OptionSeqSchema& schema) {
	auto seq = std::make_shared<AST::OptionSeq>();
	while (true) {
		Token next_token = LT(1);
		if (next_token.type() != Token::category::id) {
			break;
		}
		auto it = schema.find(next_token.value());
		if (it == schema.end()) {
			break;
		}
		auto option = std::make_shared<AST::Option>(eat(Token::category::id));
		option->value = it->second();
		seq->options.push_back(std::move(option));
	}
	return seq;
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

std::shared_ptr<AST::Block<AST::Stmt>> Parser::stmt_block() {
	Token lbrace = eat(Token::category::lbrace);

	newline_list();
	std::vector<std::shared_ptr<Stmt>> stmts;

	while (test_stmt()) {
		auto st = stmt();
		stmts.push_back(st);
		newline_list();
	}

	Token rbrace = eat(Token::category::rbrace);

	return std::make_shared<Block<Stmt>>(lbrace, rbrace,  stmts);
}

std::shared_ptr<Test> Parser::test() {
	std::shared_ptr<AttrBlock> attrs(nullptr);
	//To be honest, we should place attr list in a separate Node. And we will do that
	//just when it could be used somewhere else
	if (LA(1) == Token::category::lbracket) {
		attrs = attr_block({
			{"no_snapshots", {false, [&]{ return boolean(); }}},
			{"description", {false, [&]{ return string(); }}},
		});
		newline_list();
	}

	Token test = eat(Token::category::test);

	std::shared_ptr<Id> name = id();

	std::vector<std::shared_ptr<Id>> parents;

	if (LA(1) == Token::category::colon) {
 		eat(Token::category::colon);
 		newline_list();
 		parents.push_back(id());

 		while (LA(1) == Token::category::comma) {
 			eat(Token::category::comma);
 			newline_list();
 			parents.push_back(id());
 		}
	}

	newline_list();
	auto commands = command_block();
	return std::make_shared<Test>(attrs, test, name, parents, commands);
}

std::shared_ptr<MacroArg> Parser::macro_arg() {
	Token arg_name = eat(Token::category::id);

	std::shared_ptr<String> default_value = nullptr;

	if (LA(1) == Token::category::assign) {
		eat(Token::category::assign);

		default_value = string();
	}

	return std::shared_ptr<MacroArg>(new MacroArg(arg_name, default_value));
}

std::vector<Token> Parser::macro_body(const std::string& name) {
	std::vector<Token> result;

	result.push_back(eat(Token::category::lbrace));

	size_t braces_count = 1;

	while (braces_count != 0) {
		if (LA(1) == Token::category::lbrace) {
			braces_count++;
		} else if (LA(1) == Token::category::rbrace) {
			braces_count--;
		} else if (LA(1) == Token::category::eof) {
			throw std::runtime_error(std::string(LT(1).begin()) + ": Error: macro \"" + name + "\" body reached the end of file without closing \"}\"");
		}

		result.push_back(eat(LA(1)));
	}

	return result;
}

std::shared_ptr<Macro> Parser::macro() {
	Token macro = eat(Token::category::macro);

	Token name = eat(Token::category::id);
	eat(Token::category::lparen);

	std::vector<std::shared_ptr<MacroArg>> args;

	if (LA(1) == Token::category::id) {
		args.push_back(macro_arg());
	}

	while (LA(1) == Token::category::comma) {
		if (args.empty()) {
			eat(Token::category::rparen); //will cause failure
		}
		eat(Token::category::comma);
		args.push_back(macro_arg());
	}

	eat(Token::category::rparen);

	newline_list();
	auto body = macro_body(name.value());

	return std::make_shared<Macro>(macro, name, args, body);
}

std::shared_ptr<Param> Parser::param() {
	Token param_token = eat(Token::category::param);
	Token name = eat(Token::category::id);

	auto value = string();

	return std::make_shared<Param>(param_token, name, value);
}

std::shared_ptr<Attr> Parser::attr(const AttrBlockSchema& schema) {
	Token name = eat(Token::category::id);

	auto it = schema.find(name.value());
	if (it == schema.end()) {
		throw UnknownAttr(name);
	}
	const AttrDesc& desc = it->second;

	Token id = Token();

	if (desc.id_required) {
		id = eat(Token::category::id);
	}

	eat(Token::category::colon);
	newline_list();

	std::shared_ptr<AST::Node> value = desc.cb();

	return std::make_shared<Attr>(name, id, value);
}

std::shared_ptr<AttrBlock> Parser::attr_block(const AttrBlockSchema& schema) {
	Token lbrace = eat({Token::category::lbrace, Token::category::lbracket});

	newline_list();
	std::vector<std::shared_ptr<Attr>> attrs;

	while (LA(1) == Token::category::id) {
		std::shared_ptr<AST::Attr> new_attr = attr(schema);
		for (auto& x: attrs) {
			if (x->name() == new_attr->name()) {
				throw Exception(std::string(new_attr->begin()) + ": Error: duplicate attribute: \"" + new_attr->name() + "\"");
			}
		}
		attrs.push_back(new_attr);
		if ((LA(1) == Token::category::rbrace) || (LA(1) == Token::category::rbracket)) {
			break;
		}
		eat(Token::category::newline);
		newline_list();
	}

	newline_list();
	Token rbrace = LT(1);
	if (lbrace.type() == Token::category::lbrace) {
		eat(Token::category::rbrace);
	} else {
		eat(Token::category::rbracket);
	}

	return std::make_shared<AttrBlock>(lbrace, rbrace, attrs);
}

std::shared_ptr<AST::Controller> Parser::controller() {
	Token controller = eat({Token::category::machine, Token::category::flash, Token::category::network});

	auto name = id();

	newline_list();
	if (LA(1) != Token::category::lbrace) {
		throw std::runtime_error(std::string(LT(1).begin()) + ":Error: expected attribute block");
	}

	std::shared_ptr<AST::AttrBlock> block = nullptr;

	if (controller.type() == Token::category::machine) {
		block = attr_block({
			{"ram", {false, [&]{ return size(); }}},
			{"iso", {false, [&]{ return string(); }}},
			{"cpus", {false, [&]{ return number(); }}},
			{"qemu_spice_agent", {false, [&]{ return boolean(); }}},
			{"qemu_enable_usb3", {false, [&]{ return boolean(); }}},
			{"loader", {false, [&]{ return string(); }}},
			{"nic", {false, [&]{ return attr_block({
				{"attached_to", {false, [&]{ return id(); }}},
				{"attached_to_dev", {false, [&]{ return string(); }}},
				{"mac", {false, [&]{ return string(); }}},
				{"adapter_type", {false, [&]{ return string(); }}},
			}); }}},
			{"disk", {false, [&]{ return attr_block({
				{"size", {false, [&]{ return size(); }}},
				{"source", {false, [&]{ return string(); }}},
			}); }}},
			{"video", {false, [&]{ return attr_block({
				{"qemu_mode", {false, [&]{ return string(); }}}, // deprecated
				{"adapter_type", {false, [&]{ return string(); }}},
			}); }}},
			{"shared_folder", {false, [&]{ return attr_block({
				{"host_path", {false, [&]{ return string(); }}},
				{"readonly", {false, [&]{ return boolean(); }}},
			}); }}},
		});
	} else if (controller.type() == Token::category::flash) {
		block = attr_block({
			{"fs", {false, [&]{ return string(); }}},
			{"size", {false, [&]{ return size(); }}},
			{"folder", {false, [&]{ return string(); }}},
		});
	} else if (controller.type() == Token::category::network) {
		block = attr_block({
			{"mode", {false, [&]{ return string(); }}},
		});
	}

	return std::make_shared<AST::Controller>(controller, name, block);
}

std::shared_ptr<Cmd> Parser::command() {
	if (test_macro_call()) {
		return macro_call<AST::Cmd>();
	} else {
		auto entity = id();
		std::shared_ptr<Action> act = action();
		return std::make_shared<AST::RegularCmd>(entity, act);
	}
}

std::shared_ptr<Block<Cmd>> Parser::command_block() {
	Token lbrace = eat(Token::category::lbrace);

	newline_list();
	std::vector<std::shared_ptr<Cmd>> commands;

	while (test_command()) {
		commands.push_back(command());
		newline_list();
	}

	Token rbrace = eat(Token::category::rbrace);

	return std::make_shared<Block<Cmd>>(lbrace, rbrace, commands);
}

std::shared_ptr<IKeyCombination> Parser::key_combination() {
	if (test_string()) {
		return std::make_shared<AST::Unparsed<IKeyCombination>>(string());
	}

	std::vector<Token> buttons;

	do {
		buttons.push_back(eat(Token::category::id));

		if (LA(1) == Token::category::plus) {
			eat(Token::category::plus);
		}
	} while (LA(1) == Token::category::id);

	return std::make_shared<KeyCombination>(buttons);
}

std::shared_ptr<KeySpec> Parser::key_spec() {
	auto combination = key_combination();

	std::shared_ptr<Number> times = nullptr;

	if (LA(1) == Token::category::asterisk) {
		eat(Token::category::asterisk);
		times = number();
	}

	return std::shared_ptr<KeySpec>(new KeySpec(combination, times));
}

std::shared_ptr<Action> Parser::action() {
	bool delim_required = true;
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
		delim_required = false;
		action = action_block();
	} else if (LA(1) == Token::category::if_) {
		delim_required = false;
		action = if_clause();
	} else if (LA(1) == Token::category::for_) {
		delim_required = false;
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

	if (delim_required) {
		Token delim;
		if (LA(1) == Token::category::newline) {
			delim = eat(Token::category::newline);
		} else if (LA(1) == Token::category::semi) {
			delim = eat(Token::category::semi);
		} else {
			throw std::runtime_error(std::string(LT(1).begin()) +
				": Expected new line or ';'");
		}
		action = std::make_shared<ActionWithDelim>(action, delim);
	}

	return action;
}

std::shared_ptr<Empty> Parser::empty_action() {
	eat({Token::category::semi, Token::category::newline});
	return std::make_shared<Empty>();
}

std::shared_ptr<Abort> Parser::abort() {
	Token abort_token = eat(Token::category::abort);

	auto message = string();

	return std::make_shared<Abort>(abort_token, message);
}

std::shared_ptr<Print> Parser::print() {
	Token print_token = eat(Token::category::print);

	auto message = string();

	return std::make_shared<Print>(print_token, message);
}

std::shared_ptr<Type> Parser::type() {
	Token type_token = eat(Token::category::type_);

	auto text = string();
	std::shared_ptr<OptionSeq> options = option_seq({
		{"interval", [&]{ return time_interval(); }},
		{"autoswitch", [&]{ return key_combination(); }},
	});

	return std::make_shared<Type>(type_token, text, options);
}

std::shared_ptr<Wait> Parser::wait() {
	Token wait_token = eat(Token::category::wait);

	if (!test_selectable()) {
		throw std::runtime_error(std::string(LT(1).begin()) + " : Error: expexted an object to wait");
	}

	std::shared_ptr<SelectExpr> select_expression = select_expr();

	std::shared_ptr<OptionSeq> options = option_seq({
		{"interval", [&]{ return time_interval(); }},
		{"timeout", [&]{ return time_interval(); }},
	});

	return std::make_shared<Wait>(wait_token, select_expression, options);
}

std::shared_ptr<AST::Sleep> Parser::sleep() {
	Token sleep_token = eat(Token::category::sleep);

	auto timeout = time_interval();

	return std::make_shared<AST::Sleep>(sleep_token, timeout);
}

std::shared_ptr<Press> Parser::press() {
	Token press_token = eat(Token::category::press);

	std::vector<std::shared_ptr<KeySpec>> keys;
	keys.push_back(key_spec());

	while (LA(1) == Token::category::comma) {
		eat(Token::category::comma);
		keys.push_back(key_spec());
	}

	std::shared_ptr<OptionSeq> options = option_seq({
		{"interval", [&]{ return time_interval(); }},
	});

	return std::make_shared<Press>(press_token, keys, options);
}

std::shared_ptr<Hold> Parser::hold() {
	Token hold_token = eat(Token::category::hold);

	auto combination = key_combination();

	return std::make_shared<Hold>(hold_token, combination);
}

std::shared_ptr<Release> Parser::release() {
	Token release_token = eat(Token::category::release);

	std::shared_ptr<AST::IKeyCombination> combination = nullptr;
	if (LA(1) == Token::category::id) {
		combination = key_combination();
	}

	return std::make_shared<Release>(release_token, combination);
}

std::shared_ptr<AST::Mouse> Parser::mouse() {
	Token mouse_token = eat(Token::category::mouse);

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
	Token tmp = eat(Token::category::dot);
	Token name = LT(1);
	if (!Pos::is_adjacent(tmp.end(), name.begin())) {
		throw std::runtime_error(std::string(tmp.end()) + ": Error: expected a mouse specifier name");
	}
	eat(Token::category::id);
	Token lparen = eat(Token::category::lparen);

	Token arg;
	if (LA(1) != Token::category::rparen && LA(1) != Token::category::number) {
		throw std::runtime_error(std::string(LT(1).begin()) + " : Error: you can use only numbers as arguments in cursor specifiers");
	}

	if (LA(1) == Token::category::number) {
		arg = eat(Token::category::number);
	}

	Token rparen = eat(Token::category::rparen);

	return std::make_shared<MouseAdditionalSpecifier>(name, lparen, arg, rparen);
}

std::shared_ptr<MouseSelectable> Parser::mouse_selectable() {
	auto select = basic_select_expr();

	std::vector<std::shared_ptr<MouseAdditionalSpecifier>> specifiers;

	for (Pos it = select->end(); LA(1) == Token::category::dot && Pos::is_adjacent(it, LT(1).begin());) {
		auto specifier = mouse_additional_specifier();
		specifiers.push_back(specifier);
		it = specifier->end();
	}

	std::shared_ptr<OptionSeq> options = option_seq({
		{"timeout", [&]{ return time_interval(); }},
	});

	return std::make_shared<MouseSelectable>(select, specifiers, options);
}

std::shared_ptr<AST::MouseMoveClick> Parser::mouse_move_click() {
	Token event_token = eat({Token::category::click,
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
	Token event_token = eat(Token::category::hold);
	Token button = eat({Token::category::lbtn, Token::category::rbtn, Token::category::mbtn});
	return std::make_shared<MouseHold>(event_token, button);
}

std::shared_ptr<AST::MouseRelease> Parser::mouse_release() {
	Token event_token = eat(Token::category::release);
	return std::make_shared<MouseRelease>(event_token);
}

std::shared_ptr<AST::MouseWheel> Parser::mouse_wheel() {
	Token event_token = eat(Token::category::wheel);

	Token direction = LT(1);
	if (direction.value() != "up" && direction.value() != "down") {
		throw std::runtime_error(std::string(direction.begin()) + " : Error: unknown wheel direction: " + direction.value());
	}
	eat(Token::category::id);

	return std::make_shared<MouseWheel>(event_token, direction);
}

std::shared_ptr<MouseCoordinates> Parser::mouse_coordinates() {
	auto dx = eat(Token::category::number);
	auto dy = eat(Token::category::number);
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
	Token flash_token = eat(Token::category::flash);
	auto name = id();
	return std::make_shared<AST::PlugFlash>(flash_token, name);
}

std::shared_ptr<AST::PlugNIC> Parser::plug_resource_nic() {
	Token nic_token = eat(Token::category::id);
	auto name = id();
	return std::make_shared<AST::PlugNIC>(nic_token, name);
}

std::shared_ptr<AST::PlugLink> Parser::plug_resource_link() {
	Token link_token = eat(Token::category::id);
	auto name = id();
	return std::make_shared<AST::PlugLink>(link_token, name);
}

std::shared_ptr<AST::PlugDVD> Parser::plug_resource_dvd() {
	Token dvd_token = eat(Token::category::dvd);

	std::shared_ptr<AST::String> path = nullptr;

	if (test_string()) {
		path = string();
	}

	return std::make_shared<AST::PlugDVD>(dvd_token, path);
}

std::shared_ptr<AST::PlugHostDev> Parser::plug_resource_hostdev() {
	Token hostdev_token = eat(Token::category::hostdev);

	if (LA(1) != Token::category::usb) {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: Unknown usb device type for plug/unplug: " + LT(1).value());
	}

	Token type = eat(Token::category::usb);
	std::shared_ptr<AST::String> addr = string();

	return std::make_shared<AST::PlugHostDev>(hostdev_token, type, addr);
}

std::shared_ptr<Plug> Parser::plug() {
	Token plug_token = eat({Token::category::plug, Token::category::unplug});
	auto resource = plug_resource();
	return std::make_shared<Plug>(plug_token, resource);
}

std::shared_ptr<Start> Parser::start() {
	Token start_token = eat(Token::category::start);
	return std::make_shared<Start>(start_token);
}

std::shared_ptr<Stop> Parser::stop() {
	Token stop_token = eat(Token::category::stop);
	return std::make_shared<Stop>(stop_token);
}

std::shared_ptr<Shutdown> Parser::shutdown() {
	Token shutdown_token = eat(Token::category::shutdown);

	std::shared_ptr<OptionSeq> options = option_seq({
		{"timeout", [&]{ return time_interval(); }},
	});

	return std::make_shared<Shutdown>(shutdown_token, options);
}

std::shared_ptr<Exec> Parser::exec() {
	Token exec_token = eat(Token::category::exec);
	Token process_token = eat(Token::category::id);

	auto commands = string();

	std::shared_ptr<OptionSeq> options = option_seq({
		{"timeout", [&]{ return time_interval(); }},
	});

	return std::make_shared<Exec>(exec_token, process_token, commands, options);
}

std::shared_ptr<Copy> Parser::copy() {
	Token copy_token = eat({Token::category::copyto, Token::category::copyfrom});

	auto from = string();
	auto to = string();

	std::shared_ptr<OptionSeq> options = option_seq({
		{"nocheck", [&]{ return nullptr; }},
		{"timeout", [&]{ return time_interval(); }},
	});

	return std::make_shared<Copy>(copy_token, from, to, options);
}

std::shared_ptr<Screenshot> Parser::screenshot() {
	Token screenshot_token = eat(Token::category::screenshot);
	auto destination = string();
	return std::make_shared<Screenshot>(screenshot_token, destination);
}

std::shared_ptr<Block<Action>> Parser::action_block() {
	Token lbrace = eat(Token::category::lbrace);

	newline_list();
	std::vector<std::shared_ptr<Action>> actions;

	while (test_action()) {
		auto act = action();
		actions.push_back(act);
		newline_list();
	}

	Token rbrace = eat(Token::category::rbrace);

	return std::make_shared<Block<Action>>(lbrace, rbrace,  actions);
}

template <typename BaseType>
std::shared_ptr<MacroCall<BaseType>> Parser::macro_call() {
	Token macro_name = eat(Token::category::id);

	auto lparen = eat(Token::category::lparen);

	std::vector<std::shared_ptr<String>> params;

	if (test_string()) {
		params.push_back(string());
	}

	while (LA(1) == Token::category::comma) {
		if (params.empty()) {
			eat(Token::category::rparen); //will cause failure
		}
		eat(Token::category::comma);
		params.push_back(string());
	}

	auto rparen = eat(Token::category::rparen);
	return std::make_shared<MacroCall<BaseType>>(macro_name, lparen, params, rparen);
}

std::shared_ptr<IfClause> Parser::if_clause() {
	Token if_token = eat(Token::category::if_);
	Token open_paren = eat(Token::category::lparen);
	auto expression = expr();
	Token close_paren = eat(Token::category::rparen);

	newline_list();

	auto if_action = action();

	newline_list();
	Token else_token = Token();
	std::shared_ptr<Action> else_action = nullptr;

	if (LA(1) == Token::category::else_) {
		else_token = eat(Token::category::else_);
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
	Token range_token = eat(Token::category::RANGE);

	std::shared_ptr<Number> r1 = number();
	std::shared_ptr<Number> r2 = nullptr;
	if (test_string() || LA(1) == Token::category::number) {
		r2 = number();
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
	Token for_token = eat(Token::category::for_);
	eat(Token::category::lparen);
	Token counter = eat(Token::category::id);
	eat(Token::category::IN_);

	if (!test_counter_list()) {
		throw std::runtime_error(std::string(LT(1).begin()) + " : Error: expexted a RANGE");
	}

	std::shared_ptr<CounterList> list = counter_list();
	eat(Token::category::rparen);
	newline_list();

	auto cycle_body = action();

	Token else_token = Token();
	std::shared_ptr<Action> else_action = nullptr;

	if (LA(1) == Token::category::else_) {
		else_token = eat(Token::category::else_);
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
	Token control_token = eat({Token::category::break_, Token::category::continue_});
	return std::make_shared<CycleControl>(control_token);
}

std::shared_ptr<SelectExpr> Parser::select_expr() {
	std::shared_ptr<SelectExpr> left = select_simple_expr();

	if ((LA(1) == Token::category::double_ampersand) ||
		(LA(1) == Token::category::double_vertical_bar)) {
		return select_binop(left);
	} else {
		return left;
	}
}

std::shared_ptr<SelectSimpleExpr> Parser::select_simple_expr() {
	if (LA(1) == Token::category::exclamation_mark) {
		Token not_token = eat(Token::category::exclamation_mark);
		return std::make_shared<SelectNegationExpr>(not_token, select_simple_expr());
	} else if(LA(1) == Token::category::lparen) {
		return select_parented_expr();
	} else {
		return basic_select_expr();
	}
}

std::shared_ptr<AST::BasicSelectExpr> Parser::basic_select_expr() {
	if (test_string()) {
		return select_text();
	} else if(LA(1) == Token::category::js) {
		return select_js();
	} else if(LA(1) == Token::category::img) {
		return select_img();
	} else if(LA(1) == Token::category::homm3) {
		return select_homm3();
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + ":Error: Unknown selective object type: " + LT(1).value());
	}
}

std::shared_ptr<AST::SelectParentedExpr> Parser::select_parented_expr() {
	auto lparen = eat(Token::category::lparen);
	auto expression = select_expr();
	auto rparen = eat(Token::category::rparen);
	return std::shared_ptr<AST::SelectParentedExpr>(new AST::SelectParentedExpr(lparen, expression, rparen));
}

std::shared_ptr<SelectBinOp> Parser::select_binop(std::shared_ptr<SelectExpr> left) {
	auto op = eat({Token::category::double_ampersand, Token::category::double_vertical_bar});
	newline_list();
	auto right = select_expr();
	return std::make_shared<AST::SelectBinOp>(left, op, right);
}

std::shared_ptr<SelectJS> Parser::select_js() {
	Token js = eat(Token::category::js);
	auto script = string();
	return std::shared_ptr<SelectJS>(new SelectJS(js, script));
}

std::shared_ptr<SelectImg> Parser::select_img() {
	Token img = eat(Token::category::img);
	auto img_path = string();
	return std::shared_ptr<SelectImg>(new SelectImg(img, img_path));
}

std::shared_ptr<SelectHomm3> Parser::select_homm3() {
	Token homm3 = eat(Token::category::homm3);
	auto id = string();
	return std::shared_ptr<SelectHomm3>(new SelectHomm3(homm3, id));
}

std::shared_ptr<SelectText> Parser::select_text() {
	auto text = string();
	return std::shared_ptr<SelectText>(new SelectText({}, text));
}

std::shared_ptr<String> Parser::string() {
	if (!test_string()) {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: expected string");
	}

	Token str = eat({Token::category::quoted_string, Token::category::triple_quoted_string});

	auto new_node = std::make_shared<String>(str);

	try {
		template_literals::Parser templ_parser;
		templ_parser.validate_sanity(new_node->text());
	} catch (const std::runtime_error& error) {
		std::throw_with_nested(std::runtime_error(std::string(new_node->begin()) + ": Error parsing string: \"" + new_node->text() + "\""));
	}

	return new_node;
}

std::shared_ptr<ParentedExpr> Parser::parented_expr() {
	auto lparen = eat(Token::category::lparen);
	auto expression = expr();
	auto rparen = eat(Token::category::rparen);
	return std::shared_ptr<AST::ParentedExpr>(new AST::ParentedExpr(lparen, expression, rparen));
}

std::shared_ptr<Comparison> Parser::comparison() {
	auto left = string();

	Token op = eat({
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
	auto defined_token = eat(Token::category::DEFINED);
	Token var = eat(Token::category::id);
	return std::shared_ptr<Defined>(new Defined(defined_token, var));
}

std::shared_ptr<Check> Parser::check() {
	Token check_token = eat(Token::category::check);

	std::shared_ptr<SelectExpr> select_expression(nullptr);

	if (!test_selectable()) {
		throw std::runtime_error(std::string(LT(1).begin()) + " : Error: expexted an object to check");
	}

	select_expression = select_expr();

	std::shared_ptr<OptionSeq> options = option_seq({
		{"interval", [&]{ return time_interval(); }},
		{"timeout", [&]{ return time_interval(); }},
	});

	return std::make_shared<Check>(check_token, select_expression, options);
}

std::shared_ptr<BinOp> Parser::binop(std::shared_ptr<Expr> left) {
	auto op = eat({Token::category::OR, Token::category::AND});
	newline_list();
	auto right = expr();
	return std::make_shared<BinOp>(op, left, right);
}

std::shared_ptr<Negation> Parser::negation() {
	auto not_token = eat(Token::category::NOT);
	return std::make_shared<Negation>(not_token, simple_expr());
}

std::shared_ptr<SimpleExpr> Parser::simple_expr() {
	if (LA(1) == Token::category::NOT) {
		return negation();
	} else if (LA(1) == Token::category::check) {
		return check();
	} else if(test_defined()) {
		return defined();
	} else if (LA(1) == Token::category::lparen) {
		return parented_expr();
	} else if (test_string()) {
		return std::make_shared<StringExpr>(string());
	} else {
		throw std::runtime_error(std::string(LT(1).begin()) + ": Error: Unknown expression: " + LT(1).value());
	}
}

std::shared_ptr<Expr> Parser::expr() {
	std::shared_ptr<Expr> left = nullptr;

	if (test_comparison()) {
		left = comparison();
	} else {
		left = simple_expr();
	}

	if ((LA(1) == Token::category::AND) ||
		(LA(1) == Token::category::OR)) {
		return binop(left);
	} else {
		return left;
	}
}

std::shared_ptr<AST::Number> Parser::number() {
	return single_token<Token::category::number>();
}

std::shared_ptr<AST::Id> Parser::id() {
	return single_token<Token::category::id>();
}

std::shared_ptr<AST::TimeInterval> Parser::time_interval() {
	return single_token<Token::category::time_interval>();
}

std::shared_ptr<AST::Size> Parser::size() {
	return single_token<Token::category::size>();
}

std::shared_ptr<AST::Boolean> Parser::boolean() {
	return single_token<Token::category::boolean>();
}

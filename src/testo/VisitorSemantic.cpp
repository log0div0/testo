
#include "VisitorSemantic.hpp"
#include "Lexer.hpp"
#include "coro/Finally.h"
#include "js/Context.hpp"
#include <fmt/format.h>
#include <wildcards.hpp>

VisitorSemantic::VisitorSemantic(std::shared_ptr<Register> reg, const nlohmann::json& config): reg(reg) {
	prefix = config.at("prefix").get<std::string>();
	test_spec = config.at("test_spec").get<std::string>();
	exclude = config.at("exclude").get<std::string>();

	keys.insert("ESC");
	keys.insert("ONE");
	keys.insert("TWO");
	keys.insert("THREE");
	keys.insert("FOUR");
	keys.insert("FIVE");
	keys.insert("SIX");
	keys.insert("SEVEN");
	keys.insert("EIGHT");
	keys.insert("NINE");
	keys.insert("ZERO");
	keys.insert("MINUS");
	keys.insert("EQUALSIGN");
	keys.insert("BACKSPACE");
	keys.insert("TAB");
	keys.insert("Q");
	keys.insert("W");
	keys.insert("E");
	keys.insert("R");
	keys.insert("T");
	keys.insert("Y");
	keys.insert("U");
	keys.insert("I");
	keys.insert("O");
	keys.insert("P");
	keys.insert("LEFTBRACE");
	keys.insert("RIGHTBRACE");
	keys.insert("ENTER");
	keys.insert("LEFTCTRL");
	keys.insert("A");
	keys.insert("S");
	keys.insert("D");
	keys.insert("F");
	keys.insert("G");
	keys.insert("H");
	keys.insert("J");
	keys.insert("K");
	keys.insert("L");
	keys.insert("SEMICOLON");
	keys.insert("APOSTROPHE");
	keys.insert("GRAVE");
	keys.insert("LEFTSHIFT");
	keys.insert("BACKSLASH");
	keys.insert("Z");
	keys.insert("X");
	keys.insert("C");
	keys.insert("V");
	keys.insert("B");
	keys.insert("N");
	keys.insert("M");
	keys.insert("COMMA");
	keys.insert("DOT");
	keys.insert("SLASH");
	keys.insert("RIGHTSHIFT");
	keys.insert("LEFTALT");
	keys.insert("SPACE");
	keys.insert("CAPSLOCK");
	keys.insert("F1"),
	keys.insert("F2"),
	keys.insert("F3"),
	keys.insert("F4"),
	keys.insert("F5"),
	keys.insert("F6"),
	keys.insert("F7"),
	keys.insert("F8"),
	keys.insert("F9"),
	keys.insert("F10"),
	keys.insert("F11"),
	keys.insert("F12"),
	keys.insert("NUMLOCK");
	keys.insert("KP_0");
	keys.insert("KP_1");
	keys.insert("KP_2");
	keys.insert("KP_3");
	keys.insert("KP_4");
	keys.insert("KP_5");
	keys.insert("KP_6");
	keys.insert("KP_7");
	keys.insert("KP_8");
	keys.insert("KP_9");
	keys.insert("KP_PLUS");
	keys.insert("KP_MINUS");
	keys.insert("KP_SLASH");
	keys.insert("KP_ASTERISK");
	keys.insert("KP_ENTER");
	keys.insert("KP_DOT");
	keys.insert("SCROLLLOCK");
	keys.insert("RIGHTCTRL");
	keys.insert("RIGHTALT");
	keys.insert("HOME");
	keys.insert("UP");
	keys.insert("PAGEUP");
	keys.insert("LEFT");
	keys.insert("RIGHT");
	keys.insert("END");
	keys.insert("DOWN");
	keys.insert("PAGEDOWN");
	keys.insert("INSERT");
	keys.insert("DELETE");
	keys.insert("SCROLLUP");
	keys.insert("SCROLLDOWN");
	keys.insert("LEFTMETA");
	keys.insert("RIGHTMETA");

	//init attr ctx
	attr_ctx vm_global_ctx;
	vm_global_ctx.insert({"ram", std::make_pair(false, Token::category::size)});
	vm_global_ctx.insert({"iso", std::make_pair(false, Token::category::quoted_string)});
	vm_global_ctx.insert({"nic", std::make_pair(true, Token::category::attr_block)});
	vm_global_ctx.insert({"disk", std::make_pair(true, Token::category::attr_block)});
	vm_global_ctx.insert({"cpus", std::make_pair(false, Token::category::number)});
	vm_global_ctx.insert({"vbox_os_type", std::make_pair(false, Token::category::quoted_string)});

	attr_ctxs.insert({"vm_global", vm_global_ctx});

	attr_ctx disk_ctx;
	disk_ctx.insert({"size", std::make_pair(false, Token::category::size)});
	disk_ctx.insert({"source", std::make_pair(false, Token::category::quoted_string)});
	
	attr_ctxs.insert({"disk", disk_ctx});

	attr_ctx vm_network_ctx;
	vm_network_ctx.insert({"slot", std::make_pair(false, Token::category::number)});
	vm_network_ctx.insert({"attached_to", std::make_pair(false, Token::category::quoted_string)});
	vm_network_ctx.insert({"mac", std::make_pair(false, Token::category::quoted_string)});
	vm_network_ctx.insert({"adapter_type", std::make_pair(false, Token::category::quoted_string)});

	attr_ctxs.insert({"nic", vm_network_ctx});

	attr_ctx fd_global_ctx;
	fd_global_ctx.insert({"fs", std::make_pair(false, Token::category::quoted_string)});
	fd_global_ctx.insert({"size", std::make_pair(false, Token::category::size)});
	fd_global_ctx.insert({"folder", std::make_pair(false, Token::category::quoted_string)});

	attr_ctxs.insert({"fd_global", fd_global_ctx});

	attr_ctx network_global_ctx;
	network_global_ctx.insert({"mode", std::make_pair(false, Token::category::quoted_string)});
	network_global_ctx.insert({"persistent", std::make_pair(false, Token::category::binary)});
	network_global_ctx.insert({"autostart", std::make_pair(false, Token::category::binary)});

	attr_ctxs.insert({"network_global", network_global_ctx});

	attr_ctx test_global_ctx;
	test_global_ctx.insert({"no_snapshots", std::make_pair(false, Token::category::binary)});
	test_global_ctx.insert({"description", std::make_pair(false, Token::category::quoted_string)});
	attr_ctxs.insert({"test_global", test_global_ctx});

	testo_timeout_params.insert("TESTO_WAIT_DEFAULT_TIMEOUT");
	testo_timeout_params.insert("TESTO_WAIT_DEFAULT_INTERVAL");
	testo_timeout_params.insert("TESTO_CHECK_DEFAULT_TIMEOUT");
	testo_timeout_params.insert("TESTO_CHECK_DEFAULT_INTERVAL");
	testo_timeout_params.insert("TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT");
	testo_timeout_params.insert("TESTO_PRESS_DEFAULT_INTERVAL");
	testo_timeout_params.insert("TESTO_TYPE_DEFAULT_INTERVAL");
	testo_timeout_params.insert("TESTO_EXEC_DEFAULT_TIMEOUT");
	testo_timeout_params.insert("TESTO_COPY_DEFAULT_TIMEOUT");

	for (auto param: config.at("params")) {
		auto name = param.at("name").get<std::string>();
		auto value = param.at("value").get<std::string>();

		if (reg->params.find(name) != reg->params.end()) {
			throw std::runtime_error("Error: param \"" + name +
				"\" already exists");
		}

		auto node_found = reg->param_nodes.find(name);

		if (node_found != reg->param_nodes.end()) {
			throw std::runtime_error("Error: param with name \"" + name +
				"\" is already defined here: " + std::string((*node_found).second->begin()));
		}

		if (testo_timeout_params.find(name) != testo_timeout_params.end()) {
			if (!check_if_time_interval(value)) {
				throw std::runtime_error("Can't convert parameter " + name + " value " + value + " to time interval");
			}
		}

		if (!reg->params.insert({name, value}).second) {
			throw std::runtime_error("Error: while registering param with name " + name);
		}
	}
}

static uint32_t size_to_mb(const std::string& size) {
	uint32_t result = std::stoul(size.substr(0, size.length() - 2));
	if (size[size.length() - 2] == 'M') {
		result = result * 1;
	} else if (size[size.length() - 2] == 'G') {
		result = result * 1024;
	} else {
		throw std::runtime_error("Unknown size specifier"); //should not happen ever
	}

	return result;
}

void VisitorSemantic::visit_string_token_union(std::shared_ptr<AST::StringTokenUnion> stu) {
	auto value = template_parser.resolve(stu->text(), reg);

	Lexer lex(".", value);
	try {
		if (lex.get_next_token().type() != stu->expected_token_type) {
			throw std::runtime_error("");
		}
	} catch(const std::exception& error) {
		throw std::runtime_error(std::string(stu->begin()) + ": Error: can't convert string value \"" + value + 
			"\" to " + Token::type_to_string(stu->expected_token_type));
	}
	

	auto t = lex.get_next_token();
	if (t.type() != Token::category::eof) {
		throw std::runtime_error(std::string(stu->begin()) + ": Error: unexpected junk at the end of string: " + t.value());
	}
}

void VisitorSemantic::setup_macros_action_block(std::shared_ptr<AST::ActionBlock> action_block) {
	for (auto action: action_block->actions) {
		setup_macros_action(action);
	}
}

void VisitorSemantic::setup_macros_macro_action_call(std::shared_ptr<AST::MacroActionCall> macro_action_call) {
	auto macro_action = reg->macros_action.find(macro_action_call->name().value());
	if (macro_action == reg->macros_action.end()) {
		throw std::runtime_error(std::string(macro_action_call->begin()) + ": Error: unknown macro_action: " + macro_action_call->name().value());
	}
	macro_action_call->macro_action = macro_action->second;

	setup_macros_action_block(macro_action_call->macro_action->action_block->action);
}

void VisitorSemantic::setup_macros_if_clause(std::shared_ptr<AST::IfClause> if_clause) {
	setup_macros_action(if_clause->if_action);
	if (if_clause->has_else()) {
		setup_macros_action(if_clause->else_action);
	}
}

void VisitorSemantic::setup_macros_for_clause(std::shared_ptr<AST::ForClause> for_clause) {
	setup_macros_action(for_clause->cycle_body);

	if (for_clause->else_token) {
		setup_macros_action(for_clause->else_action);
	}
}

void VisitorSemantic::setup_macros_action(std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		return setup_macros_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroActionCall>>(action)) {
		return setup_macros_macro_action_call(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		return setup_macros_if_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		return setup_macros_for_clause(p->action);
	}
}

void VisitorSemantic::setup_macros(std::shared_ptr<AST::Test> test) {
	for (auto cmd: test->cmd_block->commands) {
		setup_macros_action(cmd->action);
	}
}

void VisitorSemantic::setup_test(std::shared_ptr<AST::Test> test) {
	if (test->is_initialized) {
		return;
	}

	for (auto parent_token: test->parents_tokens) {
		auto parent = reg->tests.find(parent_token.value());
		if (parent == reg->tests.end()) {
			throw std::runtime_error(std::string(parent_token.begin()) + ": Error: unknown test: " + parent_token.value());
		}

		for (auto already_included: test->parents) {
			if (already_included == parent->second) {
				throw std::runtime_error(std::string(parent_token.begin()) + ": Error: this test was already specified in parent list " + parent_token.value());
			}
		}

		if (parent_token.value() == test->name.value()) {
			throw std::runtime_error(std::string(parent_token.begin()) + ": Error: can't specify test as a parent to itself " + parent_token.value());
		}

		test->parents.push_back(parent->second);
	}

	setup_macros(test);

	for (auto parent: test->parents) {
		setup_test(parent);
	}

	test->is_initialized = true;
}

void VisitorSemantic::setup_tests(std::shared_ptr<AST::Program> program) {
	for (auto stmt: program->stmts) {
		if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Test>>(stmt)) {
			auto test = p->stmt;

			if (test_spec.length() && !wildcards::match(test->name.value(), test_spec)) {
				continue;
			}

			if (exclude.length() && wildcards::match(test->name.value(), exclude)) {
				continue;
			}
			setup_test(test);
		}
	}
}


void VisitorSemantic::setup_vars(std::shared_ptr<AST::Program> program) {
	for (auto stmt: program->stmts) {
		if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Test>>(stmt)) {
			auto test = p->stmt;
			//invalidate tests at request
			//So for every test
			//we need to check if it's suitable for test spec
			//if it is - push back to list and remove all the parents duplicates

			if (test_spec.length() && !wildcards::match(test->name.value(), test_spec)) {
				continue;
			}

			if (exclude.length() && wildcards::match(test->name.value(), exclude)) {
				continue;
			}
			concat_unique(tests_queue, get_test_path(test, reg));
		}
	}
}

void VisitorSemantic::visit(std::shared_ptr<AST::Program> program) {
	setup_tests(program);
	setup_vars(program);
	for (auto stmt: program->stmts) {
		visit_stmt(stmt);
	}
}

void VisitorSemantic::visit_stmt(std::shared_ptr<AST::IStmt> stmt) {
	if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Test>>(stmt)) {
		return visit_test(p->stmt);
	} else if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::MacroAction>>(stmt)) {
		return visit_macro_action(p->stmt);
	} else if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Param>>(stmt)) {
		return visit_param(p->stmt);
	} else if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Controller>>(stmt)) {
		return visit_controller(p->stmt);
	} else {
		throw std::runtime_error("Unknown statement");
	}
}

void VisitorSemantic::visit_macro_action(std::shared_ptr<AST::MacroAction> macro_action) {
	for (size_t i = 0; i < macro_action->args.size(); ++i) {
		for (size_t j = i + 1; j < macro_action->args.size(); ++j) {
			if (macro_action->args[i]->name() == macro_action->args[j]->name()) {
				throw std::runtime_error(std::string(macro_action->args[j]->begin()) + ": Error: duplicate macro arg: " + macro_action->args[j]->name());
			}
		}
	}

	bool has_default = false;
	for (auto arg: macro_action->args) {
		if (arg->default_value) {
			has_default = true;
			continue;
		}

		if (has_default && !arg->default_value) {
			throw std::runtime_error(std::string(arg->begin()) + ": Error: default value must be specified for macro arg " + arg->name());
		}
	}
}

void VisitorSemantic::visit_param(std::shared_ptr<AST::Param> param) {
	auto value = template_parser.resolve(param->value->text(), reg);

	if (testo_timeout_params.find(param->name) != testo_timeout_params.end()) {
		if (!check_if_time_interval(value)) {
			throw std::runtime_error(std::string(param->begin()) + ": Error: can't convert parameter " + param->name.value() + " value " + value + " to time interval");
		}
	}

	if (!reg->params.insert({param->name.value(), value}).second) {
		throw std::runtime_error(std::string(param->begin()) + ": Error while registering param with name " +
			param->name.value());
	}
}

void VisitorSemantic::visit_test(std::shared_ptr<AST::Test> test) {
	if (std::find(tests_queue.begin(), tests_queue.end(), test) == tests_queue.end()) {
		return;
	}

	//Check for duplicates in attrs
	nlohmann::json attrs = nlohmann::json::object();

	if (test->attrs) {
		attrs = visit_attr_block(test->attrs, "test_global");
	}

	if (attrs.count("no_snapshots")) {
		if (attrs.at("no_snapshots").get<bool>()) {
			test->snapshots_needed = false;
		}
	}

	if (attrs.count("description")) {
		test->description = attrs.at("description").get<std::string>();
	}

	visit_command_block(test->cmd_block);

	//Now that we've checked that all commands are ligit we could check that
	//all parents have totally separate vms. We can't do that before command block because
	//a user may specify unexisting vmc in some command and we need to catch that before that hierarchy check

	std::vector<std::set<std::shared_ptr<VmController>>> parents_subtries;

	//populate our parents paths
	for (auto parent: test->parents) {
		parents_subtries.push_back(get_all_vmcs(parent, reg));
	}

	//check that parents path are independent
	for (size_t i = 0; i < parents_subtries.size(); ++i) {
		for (size_t j = 0; j < parents_subtries.size(); ++j) {
			if (i == j) {
				continue;
			}

			std::vector<std::shared_ptr<VmController>> intersection;

			std::set_intersection(
				parents_subtries[i].begin(), parents_subtries[i].end(),
				parents_subtries[j].begin(), parents_subtries[j].end(),
				std::back_inserter(intersection));

			if (intersection.size() != 0) {
				throw std::runtime_error(std::string(test->begin()) + ":Error: some parents have common vms");
			}
		}
	}
}

void VisitorSemantic::visit_command_block(std::shared_ptr<AST::CmdBlock> block) {
	for (auto command: block->commands) {
		visit_command(command);
	}
}

void VisitorSemantic::visit_command(std::shared_ptr<AST::Cmd> cmd) {
	std::set<std::shared_ptr<VmController>> unique_vmcs;
	for (auto vm_token: cmd->vms) {
		auto vmc_request = reg->vmc_requests.find(vm_token.value());
		if (vmc_request == reg->vmc_requests.end()) {
			throw std::runtime_error(std::string(vm_token.begin()) + ": Error: unknown vitrual machine name: " + vm_token.value());
		}

		if (!unique_vmcs.insert(vmc_request->second.get_vmc()).second) {
			throw std::runtime_error(std::string(vm_token.begin()) + ": Error: this vmc was already specified in the virtual machines list: " + vm_token.value());
		}
	}

	visit_action(cmd->action);
}

void VisitorSemantic::visit_action_block(std::shared_ptr<AST::ActionBlock> action_block) {
	for (auto action: action_block->actions) {
		visit_action(action);
	}
}

void VisitorSemantic::visit_action(std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Press>>(action)) {
		return visit_press(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Type>>(action)) {
		if (p->action->interval) {
			return visit_string_token_union(p->action->interval);
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Hold>>(action)) {
		return visit_key_combination(p->action->combination);
	}  else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Release>>(action)) {
		if (p->action->combination) {
			return visit_key_combination(p->action->combination);
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		return visit_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Mouse>>(action)) {
		return visit_mouse(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Plug>>(action)) {
		return visit_plug(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		return visit_string_token_union(p->action->timeout);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Shutdown>>(action)) {
		if (p->action->timeout) {
			return visit_string_token_union(p->action->timeout);
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Exec>>(action)) {
		return visit_exec(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Copy>>(action)) {
		if (p->action->timeout) {
			return visit_string_token_union(p->action->timeout);
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Wait>>(action)) {
		return visit_wait(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroActionCall>>(action)) {
		return visit_macro_action_call(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		return visit_if_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		return visit_for_clause(p->action);
	} 
}

void VisitorSemantic::visit_press(std::shared_ptr<AST::Press> press) {
	for (auto key_spec: press->keys) {
		visit_key_spec(key_spec);
	}
	
	if (press->interval) {
		visit_string_token_union(press->interval);
	}
}

void VisitorSemantic::visit_key_combination(std::shared_ptr<AST::KeyCombination> combination) {
	for (size_t i = 0; i < combination->buttons.size(); ++i) {
		auto button = combination->buttons[i];
		if (!is_button(button)) {
			throw std::runtime_error(std::string(button.begin()) +
				" :Error: Unknown key " + button.value());
		}

		for (size_t j = i + 1; j < combination->buttons.size(); ++j) {
			if (button.value() == combination->buttons[j].value()) {
				throw std::runtime_error(std::string(combination->buttons[j].begin()) +
					" :Error: duplicate key " + button.value());
			}
		}
	}
}

void VisitorSemantic::visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec) {
	visit_key_combination(key_spec->combination);

	if (key_spec->times.value().length()) {
		if (std::stoi(key_spec->times.value()) < 1) {
			throw std::runtime_error(std::string(key_spec->times.begin()) +
					" :Error: Can't press a button less than 1 time: " + key_spec->times.value());
		}
	}
}


void VisitorSemantic::visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers)
{
	//finally we're here

	/*
	what checks do we need?
	1) If we have from, there could not be another from
	2) If we have center, there could not be another center
	3) From could not be after center or move
	4) Center could not be after move
	This should cover it
	*/

	bool has_from = false;
	bool has_center = false;
	bool has_move = false;

	for (auto specifier: specifiers) {
		auto arg = specifier->arg;

		if (specifier->is_from()) {
			if (!arg) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: specifier " + specifier->name.value() + " requires a number as an argument");
			}

			try {
				std::stoi(arg.value());
			} catch (const std::exception& error) {
				throw std::runtime_error(std::string(arg.begin()) + ": Error: the argument must be a number");
			}

			if (has_from) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after another \"from\" specifier");
			}
			if (has_center) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after a \"precision\" specifier");
			}
			if (has_move) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after a \"move\" specifier");
			}
			has_from = true;
			continue;
		} if (specifier->is_centering()) {
			if (arg) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: specifier " + specifier->name.value() + " must not have an argument");
			}
			if (has_center) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after another \"precision\" specifier");
			}
			if (has_move) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after a \"move\" specifier");
			}
			has_center = true;
			continue;
		} else if (specifier->is_moving()) {
			if (!arg) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: specifier " + specifier->name.value() + " requires a number as an argument");
			}
			try {
				std::stoi(arg.value());
			} catch (const std::exception& error) {
				throw std::runtime_error(std::string(arg.begin()) + ": Error: the argument must be a number");
			}
			has_move = true;
			continue;
		} else {
			throw std::runtime_error(std::string(specifier->begin()) + ": Error: unknown specifier: " + specifier->name.value());
		}
	}
}

void VisitorSemantic::visit_mouse_move_selectable(std::shared_ptr<AST::MouseSelectable> mouse_selectable) {
	if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(mouse_selectable->selectable)) {
		if (mouse_selectable->specifiers.size()) {
			throw std::runtime_error(std::string(mouse_selectable->specifiers[0]->begin()) + ": Error: mouse specifiers are not supported for js selections");
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::String>>(mouse_selectable->selectable)) {
		visit_mouse_additional_specifiers(mouse_selectable->specifiers);
	}

	if (mouse_selectable->timeout) {
		visit_string_token_union(mouse_selectable->timeout);
	}
}

void VisitorSemantic::visit_mouse_move_click(std::shared_ptr<AST::MouseMoveClick> mouse_move_click) {
	if (mouse_move_click->object) {
		if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseSelectable>>(mouse_move_click->object)) {
			visit_mouse_move_selectable(p->target);
		}
	}
}

void VisitorSemantic::visit_mouse(std::shared_ptr<AST::Mouse> mouse) {
	if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseMoveClick>>(mouse->event)) {
		return visit_mouse_move_click(p->event);
	}
}

void VisitorSemantic::visit_plug(std::shared_ptr<AST::Plug> plug) {
	if (plug->name) {
		visit_string_token_union(plug->name);
	}

	if (plug->type.value() == "flash") {
		auto name = template_parser.resolve(plug->name->text(), reg);
		if (reg->fdc_requests.find(name) == reg->fdc_requests.end()) {
			throw std::runtime_error(std::string(plug->begin()) + ": Error: Unknown flash drive: " + name);
		}
	}
}

void VisitorSemantic::visit_exec(std::shared_ptr<AST::Exec> exec) {
	if ((exec->process_token.value() != "bash") &&
		(exec->process_token.value() != "cmd") &&
		(exec->process_token.value() != "python") &&
		(exec->process_token.value() != "python2") &&
		(exec->process_token.value() != "python3"))
	{
		throw std::runtime_error(std::string(exec->begin()) + ": Error: unsupported interpreter: " + exec->process_token.value());
	}

	if (exec->timeout) {
		visit_string_token_union(exec->timeout);
	}
}

void VisitorSemantic::visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr) {
	if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::ISelectable>>(select_expr)) {
		return visit_detect_selectable(p->select_expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::SelectUnOp>>(select_expr)) {
		return visit_detect_unop(p->select_expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::SelectBinOp>>(select_expr)) {
		return visit_detect_binop(p->select_expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::SelectParentedExpr>>(select_expr)) {
		return visit_detect_expr(p->select_expr->select_expr);
	}
}

void VisitorSemantic::validate_js(const std::string& script) {
	js::Context js_ctx(nullptr);
	js_ctx.eval(script, true);
}

void VisitorSemantic::visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable) {
	std::string query = "";
	if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::String>>(selectable)) {
		auto text = template_parser.resolve(p->text(), reg);
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(selectable)) {
		auto script = template_parser.resolve(p->text(), reg);
		try {
			validate_js(script);
		} catch (const std::exception& error) {
			std::throw_with_nested(std::runtime_error(std::string(p->begin()) + ": Error while validating javascript selection"));
		}
	}
}

void VisitorSemantic::visit_detect_unop(std::shared_ptr<AST::SelectUnOp> unop) {
	visit_detect_expr(unop->select_expr);
}

void VisitorSemantic::visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop) {
	visit_detect_expr(binop->left);
	visit_detect_expr(binop->right);
}

void VisitorSemantic::visit_wait(std::shared_ptr<AST::Wait> wait) {
	visit_detect_expr(wait->select_expr);

	if (wait->timeout) {
		visit_string_token_union(wait->timeout);
	}

	if (wait->interval) {
		visit_string_token_union(wait->interval);
	}
}

void VisitorSemantic::visit_macro_action_call(std::shared_ptr<AST::MacroActionCall> macro_action_call) {
	uint32_t args_with_default = 0;

	for (auto arg: macro_action_call->macro_action->args) {
		if (arg->default_value) {
			args_with_default++;
		}
	}

	if (macro_action_call->args.size() < macro_action_call->macro_action->args.size() - args_with_default) {
		throw std::runtime_error(fmt::format("{}: Error: expected at least {} args, {} provided", std::string(macro_action_call->begin()),
			macro_action_call->macro_action->args.size() - args_with_default, macro_action_call->args.size()));
	}

	if (macro_action_call->args.size() > macro_action_call->macro_action->args.size()) {
		throw std::runtime_error(fmt::format("{}: Error: expected at most {} args, {} provided", std::string(macro_action_call->begin()),
			macro_action_call->macro_action->args.size(), macro_action_call->args.size()));
	}

	//push new ctx
	StackEntry new_ctx(true);

	for (size_t i = 0; i < macro_action_call->args.size(); ++i) {
		auto value = template_parser.resolve(macro_action_call->args[i]->text(), reg);
		new_ctx.define(macro_action_call->macro_action->args[i]->name(), value);
	}

	for (size_t i = macro_action_call->args.size(); i < macro_action_call->macro_action->args.size(); ++i) {
		auto value = template_parser.resolve(macro_action_call->macro_action->args[i]->default_value->text(), reg);
		new_ctx.define(macro_action_call->macro_action->args[i]->name(), value);
	}

	reg->local_vars.push_back(new_ctx);
	coro::Finally finally([&] {
		reg->local_vars.pop_back();
	});

	visit_action_block(macro_action_call->macro_action->action_block->action);
}

void VisitorSemantic::visit_expr(std::shared_ptr<AST::IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::BinOp>>(expr)) {
		return visit_binop(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::IFactor>>(expr)) {
		return visit_factor(p->expr);
	}
}


void VisitorSemantic::visit_binop(std::shared_ptr<AST::BinOp> binop) {
	visit_expr(binop->left);
	visit_expr(binop->right);
}

void VisitorSemantic::visit_factor(std::shared_ptr<AST::IFactor> factor) {
	if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Check>>(factor)) {
		return visit_check(p->factor);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::IExpr>>(factor)) {
		return visit_expr(p->factor);
	}
}

void VisitorSemantic::visit_check(std::shared_ptr<AST::Check> check) {
	visit_detect_expr(check->select_expr);

	if (check->timeout) {
		visit_string_token_union(check->timeout);
	}

	if (check->interval) {
		visit_string_token_union(check->interval);
	}
}

void VisitorSemantic::visit_if_clause(std::shared_ptr<AST::IfClause> if_clause) {
	visit_expr(if_clause->expr);
	visit_action(if_clause->if_action);
	if (if_clause->has_else()) {
		visit_action(if_clause->else_action);
	}
}

void VisitorSemantic::visit_range(std::shared_ptr<AST::Range> range) {
	std::string r1 = template_parser.resolve(range->r1->text(), reg);
	std::string r2;

	try {
		range->r1_num = std::stoul(r1);
	} catch (const std::exception& error) {
		std::runtime_error(std::string(range->r1->begin()) + ": Error: Can't convert to uint: " + r1);
	}

	if (range->r2) {
		r2 = template_parser.resolve(range->r2->text(), reg);
		try {
			range->r2_num = std::stoul(r2);
		} catch (const std::exception& error) {
			std::runtime_error(std::string(range->r2->begin()) + ": Error: Can't convert to uint: " + r2);
		}

		if (range->r1_num >= range->r2_num) {
			throw std::runtime_error(std::string(range->begin()) + ": Error: start of the range " +
				r1 + " is greater or equal to finish " + r2);
		}
	}
}

void VisitorSemantic::visit_for_clause(std::shared_ptr<AST::ForClause> for_clause) {
	if (auto p = std::dynamic_pointer_cast<AST::CounterList<AST::Range>>(for_clause->counter_list)) {
		visit_range(p->counter_list);
	} else {
		throw std::runtime_error("Unknown counter list type");
	}

	StackEntry new_ctx(false);
	reg->local_vars.push_back(new_ctx);
	size_t ctx_position = reg->local_vars.size() - 1;
	coro::Finally finally([&]{
		reg->local_vars.pop_back();
	});

	for (auto i: for_clause->counter_list->values()) {
		reg->local_vars[ctx_position].define(for_clause->counter.value(), i);
		visit_action(for_clause->cycle_body);
	}

	if (for_clause->else_token) {
		visit_action(for_clause->else_action);
	}
}

void VisitorSemantic::visit_controller(std::shared_ptr<AST::Controller> controller) {
	if (controller->t.type() == Token::category::machine) {
		return visit_machine(controller);
	} else if (controller->t.type() == Token::category::flash) {
		return visit_flash(controller);
	} else if (controller->t.type() == Token::category::network) {
		return visit_network(controller);
	} else {
		throw std::runtime_error("Unknown controller type");
	}
}


void VisitorSemantic::visit_machine(std::shared_ptr<AST::Controller> machine) {
	auto config = visit_attr_block(machine->attr_block, "vm_global");
	config["prefix"] = prefix;
	config["name"] = machine->name.value();
	config["src_file"] = machine->name.begin().file.generic_string();

	reg->vmc_requests.emplace(std::make_pair(machine->name, VmControllerRequest(config, reg)));
}

void VisitorSemantic::visit_flash(std::shared_ptr<AST::Controller> flash) {
	//no need to check for duplicates
	//It's already done in Parser while registering Controller
	auto config = visit_attr_block(flash->attr_block, "fd_global");
	config["prefix"] = prefix;
	config["name"] = flash->name.value();
	config["src_file"] = flash->name.begin().file.generic_string();

	reg->fdc_requests.emplace(std::make_pair(flash->name, FlashDriveControllerRequest(config)));
}

void VisitorSemantic::visit_network(std::shared_ptr<AST::Controller> network) {
	auto config = visit_attr_block(network->attr_block, "network_global");
	config["prefix"] = prefix;
	config["name"] = network->name.value();
	config["src_file"] = network->name.begin().file.generic_string();

	auto netc = env->create_network_controller(config);
	reg->netcs.emplace(std::make_pair(network->name, netc));
}

nlohmann::json VisitorSemantic::visit_attr_block(std::shared_ptr<AST::AttrBlock> attr_block, const std::string& ctx_name) {
	nlohmann::json config;
	for (auto attr: attr_block->attrs) {
		visit_attr(attr, config, ctx_name);
	}
	return config;
}

void VisitorSemantic::visit_attr(std::shared_ptr<AST::Attr> attr, nlohmann::json& config, const std::string& ctx_name) {
	auto ctx = attr_ctxs.find(ctx_name);
	if (ctx == attr_ctxs.end()) {
		throw std::runtime_error("Unknown ctx"); //should never happen
	}

	auto found = ctx->second.find(attr->name);

	if (found == ctx->second.end()) {
		throw std::runtime_error(std::string(attr->begin()) + ": Error: unknown attr name: " + attr->name.value());
	}

	auto match = found->second;
	if (attr->id != match.first) {
		if (match.first) {
			throw std::runtime_error(std::string(attr->end()) + ": Error: attribute " + attr->name.value() +
				" requires a name");
		} else {
			throw std::runtime_error(std::string(attr->end()) + ": Error: attribute " + attr->name.value() +
				" must have no name");
		}
	}

	if (attr->value->t.type() != match.second) {
		throw std::runtime_error(std::string(attr->end()) + ": Error: unexpected value type " +
			Token::type_to_string(attr->value->t.type()) + " for attr " + attr->name.value() + ", expected " +
			Token::type_to_string(match.second));
	}

	if (config.count(attr->name.value())) {
		if (!config.at(attr->name.value()).is_array()) {
			throw std::runtime_error(std::string(attr->begin()) + ": Error: duplicate attr " + attr->name.value());
		}
	}

	if (auto p = std::dynamic_pointer_cast<AST::AttrValue<AST::StringAttr>>(attr->value)) {
		auto value = template_parser.resolve(p->attr_value->value->text(), reg);
		config[attr->name.value()] = value;
	} else if (auto p = std::dynamic_pointer_cast<AST::AttrValue<AST::BinaryAttr>>(attr->value)) {
		auto value = p->attr_value->value;
		if (value.type() == Token::category::true_) {
			config[attr->name.value()] = true;
		} else if (value.type() == Token::category::false_) {
			config[attr->name.value()] = false;
		} else {
			throw std::runtime_error(std::string(attr->begin()) + ": Error: unsupported binary attr: " + value.value());
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::AttrValue<AST::SimpleAttr>>(attr->value)) {
		auto value = p->attr_value->t;
		if (value.type() == Token::category::number) {
			if (std::stoi(value.value()) < 0) {
				throw std::runtime_error(std::string(attr->begin()) + ": Error: numeric attr can't be negative: " + value.value());
			}
			config[attr->name.value()] = std::stoul(value.value());
		} else if (value.type() == Token::category::size) {
			config[attr->name.value()] = size_to_mb(value);
		} else {
			 throw std::runtime_error(std::string(attr->begin()) + ": Error: unsupported attr: " + value.value());
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::AttrValue<AST::AttrBlock>>(attr->value)) {
		//we assume for now that named attrs could be only in attr_blocks
		auto j = visit_attr_block(p->attr_value, attr->name);
		if (attr->id) {
			j["name"] = attr->id.value();
			config[attr->name.value()].push_back(j);
		}  else {
			config[attr->name.value()] = visit_attr_block(p->attr_value, attr->name);
		}
	} else {
		throw std::runtime_error("Unknown attr category");
	}
}

bool VisitorSemantic::is_button(const Token& t) const {
	std::string button = t.value();
	std::transform(button.begin(), button.end(), button.begin(), ::toupper);
	return (keys.find(button) != keys.end());
}

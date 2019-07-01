
#include "VisitorSemantic.hpp"
#include <fmt/format.h>

using namespace AST;

VisitorSemantic::VisitorSemantic(Register& reg):
	reg(reg)
{
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

		//init attr ctx
		attr_ctx vm_global_ctx;
		vm_global_ctx.insert({"ram", std::make_pair(false, Token::category::size)});
		vm_global_ctx.insert({"disk_size", std::make_pair(false, Token::category::size)});
		vm_global_ctx.insert({"iso", std::make_pair(false, Token::category::word)});
		vm_global_ctx.insert({"nic", std::make_pair(true, Token::category::attr_block)});
		vm_global_ctx.insert({"cpus", std::make_pair(false, Token::category::number)});
		vm_global_ctx.insert({"vbox_os_type", std::make_pair(false, Token::category::word)});
		vm_global_ctx.insert({"metadata", std::make_pair(false, Token::category::attr_block)});

		attr_ctxs.insert({"vm_global", vm_global_ctx});

		attr_ctx vm_network_ctx;
		vm_network_ctx.insert({"slot", std::make_pair(false, Token::category::number)});
		vm_network_ctx.insert({"attached_to", std::make_pair(false, Token::category::word)});
		vm_network_ctx.insert({"network", std::make_pair(false, Token::category::word)});
		vm_network_ctx.insert({"mac", std::make_pair(false, Token::category::word)});
		vm_network_ctx.insert({"adapter_type", std::make_pair(false, Token::category::word)});

		attr_ctxs.insert({"nic", vm_network_ctx});

		attr_ctx fd_global_ctx;
		fd_global_ctx.insert({"fs", std::make_pair(false, Token::category::word)});
		fd_global_ctx.insert({"size", std::make_pair(false, Token::category::size)});
		fd_global_ctx.insert({"folder", std::make_pair(false, Token::category::word)});
		fd_global_ctx.insert({"cache_enabled", std::make_pair(false, Token::category::number)});

		attr_ctxs.insert({"fd_global", fd_global_ctx});
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

void VisitorSemantic::update_leaves() {
	for (auto it: reg.tests) {
		bool is_leaf = true;
		for (auto it2: reg.tests) {
			for (auto parent: it2.second->parents) {
				if (it.second->name.value() == parent->name.value()) {
					is_leaf = false;
					break;
				}
			}
			if (is_leaf) {
				break;
			}
		}

		if (is_leaf) {
			it.second->snapshots_needed = false;
		}
	}
}

void VisitorSemantic::visit(std::shared_ptr<Program> program) {
	for (auto stmt: program->stmts) {
		visit_stmt(stmt);
	}

	update_leaves();
}

void VisitorSemantic::visit_stmt(std::shared_ptr<IStmt> stmt) {
	if (auto p = std::dynamic_pointer_cast<Stmt<Test>>(stmt)) {
		return visit_test(p->stmt);
	} else if (auto p = std::dynamic_pointer_cast<Stmt<Macro>>(stmt)) {
		return visit_macro(p->stmt);
	} else if (auto p = std::dynamic_pointer_cast<Stmt<Controller>>(stmt)) {
		return visit_controller(p->stmt);
	} else {
		throw std::runtime_error("Unknown statement");
	}
}

void VisitorSemantic::visit_macro(std::shared_ptr<Macro> macro) {
	std::cout << "Registering macro " << macro->name.value() << std::endl;

	if (reg.macros.find(macro->name) != reg.macros.end()) {
		throw std::runtime_error(std::string(macro->begin()) + ": Error: macros with name " + macro->name.value() +
			" already exists");
	}

	if (!reg.macros.insert({macro->name, macro}).second) {
		throw std::runtime_error(std::string(macro->begin()) + ": Error while registering macro with name " +
			macro->name.value());
	}

	visit_action_block(macro->action_block->action); //dummy controller to match the interface
}

void VisitorSemantic::visit_test(std::shared_ptr<Test> test) {
	//Check for duplicates in attrs
	std::set<std::string> attrs;

	for (auto attr: test->attrs) {
		if (!attrs.insert(attr.value()).second) {
			throw std::runtime_error(std::string(attr.pos()) + ": Error: duplicate attribute : " + attr.value());
		}
	}

	for (auto attr: attrs) {
		if (attr == "no_snapshots") {
			test->snapshots_needed = false;
		} else {
			throw std::runtime_error(std::string(test->begin()) + ": Error: unknown attribute : " + attr);
		}
	}

	for (auto parent_token: test->parents_tokens) {
		auto parent = reg.tests.find(parent_token.value());
		if (parent == reg.tests.end()) {
			throw std::runtime_error(std::string(parent_token.pos()) + ": Error: unknown test: " + parent_token.value());
		}

		for (auto already_included: test->parents) {
			if (already_included == parent->second) {
				throw std::runtime_error(std::string(parent_token.pos()) + ": Error: this test was already specified in parent list " + parent_token.value());
			}
		}

		test->parents.push_back(parent->second);

		if (parent_token.value() == test->name.value()) {
			throw std::runtime_error(std::string(parent_token.pos()) + ": Error: can't specify test as a parent to itself " + parent_token.value());
		}
	}

	if (!reg.tests.insert({test->name.value(), test}).second) {
		throw std::runtime_error(std::string(test->begin()) + ": Error test is already defined: " +
			test->name.value());
	}

	visit_command_block(test->cmd_block);

	//Now that we've checked that all commands are ligit we could check that
	//all parents have totally separate vms. We can't do that before command block because
	//a user may specify unexisting vmc in some command and we need to catch that before that hierarchy check

	std::vector<std::set<std::shared_ptr<VmController>>> parents_subtries;

	//populate our parents paths
	for (auto parent: test->parents) {
		parents_subtries.push_back(reg.get_all_vmcs(parent));
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

void VisitorSemantic::visit_command_block(std::shared_ptr<CmdBlock> block) {
	for (auto command: block->commands) {
		visit_command(command);
	}
}

void VisitorSemantic::visit_command(std::shared_ptr<Cmd> cmd) {
	std::set<std::shared_ptr<VmController>> unique_vmcs;
	for (auto vm_token: cmd->vms) {
		auto vmc = reg.vmcs.find(vm_token.value());
		if (vmc == reg.vmcs.end()) {
			throw std::runtime_error(std::string(vm_token.pos()) + ": Error: unknown vitrual machine name: " + vm_token.value());
		}

		if (!unique_vmcs.insert(vmc->second).second) {
			throw std::runtime_error(std::string(vm_token.pos()) + ": Error: this vmc was already specified in the virtual machines list: " + vm_token.value());
		}
	}

	visit_action(cmd->action);
}

void VisitorSemantic::visit_action_block(std::shared_ptr<ActionBlock> action_block) {
	for (auto action: action_block->actions) {
		visit_action(action);
	}
}

void VisitorSemantic::visit_action(std::shared_ptr<IAction> action) {
	if (auto p = std::dynamic_pointer_cast<Action<Press>>(action)) {
		return visit_press(p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<ActionBlock>>(action)) {
		return visit_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Plug>>(action)) {
		return visit_plug(p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Exec>>(action)) {
		return visit_exec(p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<MacroCall>>(action)) {
		return visit_macro_call(p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<ForClause>>(action)) {
		return visit_for_clause(p->action);
	}
}

void VisitorSemantic::visit_press(std::shared_ptr<Press> press) {
	for (auto key_spec: press->keys) {
		visit_key_spec(key_spec);
	}
}

void VisitorSemantic::visit_key_spec(std::shared_ptr<KeySpec> key_spec) {
	for (auto button: key_spec->buttons) {
		if (!is_button(button)) {
			throw std::runtime_error(std::string(button.pos()) +
				" :Error: Unknown key " + button.value());
		}
	}
}

void VisitorSemantic::visit_plug(std::shared_ptr<Plug> plug) {
	if (plug->type.value() == "flash") {
		if (reg.fds.find(plug->name_token.value()) == reg.fds.end()) {
			throw std::runtime_error(std::string(plug->begin()) + ": Error: Unknown flash drive: " + plug->name_token.value());
		}
	}
}

void VisitorSemantic::visit_exec(std::shared_ptr<Exec> exec) {
	if (exec->process_token.value() != "bash") {
		throw std::runtime_error(std::string(exec->begin()) + ": Error: unknown process name: " + exec->process_token.value());
	}
}

void VisitorSemantic::visit_macro_call(std::shared_ptr<MacroCall> macro_call) {
	auto macro = reg.macros.find(macro_call->name());
	if (macro == reg.macros.end()) {
		throw std::runtime_error(std::string(macro_call->begin()) + ": Error: unknown macro: " + macro_call->name().value());
	}
	macro_call->macro = macro->second;
	if (macro_call->params.size() != macro_call->macro->params.size()) {
		throw std::runtime_error(fmt::format("{}: Error: expected {} params, {} provided", std::string(macro_call->begin()),
			macro_call->macro->params.size(), macro_call->params.size()));
	}
}

void VisitorSemantic::visit_for_clause(std::shared_ptr<ForClause> for_clause) {
	if (for_clause->start() > for_clause->finish()) {
		throw std::runtime_error(std::string(for_clause->begin()) + ": Error: start number of the cycle " +
			for_clause->start_.value() + " is greater than finish number " + for_clause->finish_.value());
	}
}

void VisitorSemantic::visit_controller(std::shared_ptr<Controller> controller) {
	if (controller->t.type() == Token::category::machine) {
		return visit_machine(controller);
	} else if (controller->t.type() == Token::category::flash) {
		return visit_flash(controller);
	} else {
		throw std::runtime_error("Unknown controller type");
	}
}

void VisitorSemantic::visit_machine(std::shared_ptr<Controller> machine) {
	std::cout << "Registering machine " << machine->name.value() << std::endl;
	if (reg.vmcs.find(machine->name) != reg.vmcs.end()) {
		throw std::runtime_error(std::string(machine->begin()) + ": Error: machine with name " + machine->name.value() +
			" already exists");
	}

	auto config = visit_attr_block(machine->attr_block, "vm_global");
	config["name"] = machine->name.value();

	auto vmc = env->create_vm_controller(config);
	reg.vmcs.emplace(std::make_pair(machine->name, vmc));
}

void VisitorSemantic::visit_flash(std::shared_ptr<Controller> flash) {
	std::cout << "Registering flash " << flash->name.value() << std::endl;
	if (reg.fds.find(flash->name) != reg.fds.end()) {
		throw std::runtime_error(std::string(flash->begin()) + ": Error: flash drive with name " + flash->name.value() +
			" already exists");
	}

	auto config = visit_attr_block(flash->attr_block, "fd_global");
	config["name"] = flash->name.value();

	auto fd = env->create_flash_drive_controller(config);
	reg.fds.emplace(std::make_pair(flash->name, fd));
}

nlohmann::json VisitorSemantic::visit_attr_block(std::shared_ptr<AttrBlock> attr_block, const std::string& ctx_name) {
	nlohmann::json config;
	for (auto attr: attr_block->attrs) {
		visit_attr(attr, config, ctx_name);
	}
	return config;
}

std::string VisitorSemantic::resolve_var(const std::string& var) {
	auto env_value = std::getenv(var.c_str());

	if (env_value == nullptr) {
		return "";
	}
	return env_value;
}

std::string VisitorSemantic::visit_word(std::shared_ptr<Word> word) {
	std::string result;

	for (auto part: word->parts) {
		if (part.type() == Token::category::dbl_quoted_string) {
			result += part.value().substr(1, part.value().length() - 2);
		} else if (part.type() == Token::category::var_ref) {
			result += resolve_var(part.value().substr(1, part.value().length() - 1));
		} else if (part.type() == Token::category::multiline_string) {
			result += part.value().substr(3, part.value().length() - 6);
		} else {
			throw std::runtime_error("Unknown word type");
		}
	}

	return result;
}

void VisitorSemantic::visit_attr(std::shared_ptr<Attr> attr, nlohmann::json& config, const std::string& ctx_name) {
	if (ctx_name == "metadata") {
		if (attr->value->t.type() != Token::category::word) {
			throw std::runtime_error(std::string(attr->begin()) + ": Error: metadata supports only word specifiers ");
		}
	} else {
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
	}

	if (config.count(attr->name.value())) {
		if (!config.at(attr->name.value()).is_array()) {
			throw std::runtime_error(std::string(attr->begin()) + ": Error: duplicate attr " + attr->name.value());
		}
	}

	if (auto p = std::dynamic_pointer_cast<AttrValue<WordAttr>>(attr->value)) {
		auto value = visit_word(p->attr_value->value);
		config[attr->name.value()] = value;
	} else if (auto p = std::dynamic_pointer_cast<AttrValue<SimpleAttr>>(attr->value)) {
		auto value = p->attr_value->t;
		if (value.type() == Token::category::number) {
			config[attr->name.value()] = std::stoul(value.value());
		} else if (value.type() == Token::category::size) {
			config[attr->name.value()] = size_to_mb(value);
		} else {
			 throw std::runtime_error(std::string(attr->begin()) + ": Error: unsupported attr: " + value.value());
		}
	} else if (auto p = std::dynamic_pointer_cast<AttrValue<AttrBlock>>(attr->value)) {
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

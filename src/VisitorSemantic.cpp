
#include <VisitorSemantic.hpp>

using namespace AST;

VisitorSemantic::VisitorSemantic(Global& global):
	global(global) 
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
		keys.insert("EQUAL");
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
		vm_global_ctx.insert({"iso", std::make_pair(false, Token::category::dbl_quoted_string)});
		vm_global_ctx.insert({"nic", std::make_pair(true, Token::category::attr_block)});
		vm_global_ctx.insert({"cpus", std::make_pair(false, Token::category::number)});
		vm_global_ctx.insert({"os_type", std::make_pair(false, Token::category::dbl_quoted_string)});
		vm_global_ctx.insert({"metadata", std::make_pair(false, Token::category::attr_block)});

		attr_ctxs.insert({"vm_global", vm_global_ctx});

		attr_ctx vm_network_ctx;
		vm_network_ctx.insert({"slot", std::make_pair(false, Token::category::number)});
		vm_network_ctx.insert({"attached_to", std::make_pair(false, Token::category::dbl_quoted_string)});
		vm_network_ctx.insert({"network", std::make_pair(false, Token::category::dbl_quoted_string)});
		vm_network_ctx.insert({"mac", std::make_pair(false, Token::category::dbl_quoted_string)});

		attr_ctxs.insert({"nic", vm_network_ctx});

		attr_ctx fd_global_ctx;
		fd_global_ctx.insert({"fs", std::make_pair(false, Token::category::dbl_quoted_string)});
		fd_global_ctx.insert({"size", std::make_pair(false, Token::category::size)});
		fd_global_ctx.insert({"folder", std::make_pair(false, Token::category::dbl_quoted_string)});

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

void VisitorSemantic::visit(std::shared_ptr<Program> program) {
	for (auto stmt: program->stmts) {
		visit_stmt(stmt);
	}
}

void VisitorSemantic::visit_stmt(std::shared_ptr<IStmt> stmt) {
	if (auto p = std::dynamic_pointer_cast<Stmt<Snapshot>>(stmt)) {
		return visit_snapshot(p->stmt);
	} else if (auto p = std::dynamic_pointer_cast<Stmt<Test>>(stmt)) {
		return visit_test(p->stmt);
	} else if (auto p = std::dynamic_pointer_cast<Stmt<Controller>>(stmt)) {
		return visit_controller(p->stmt);
	} else {
		throw std::runtime_error("Unknown statement");
	}
}


void VisitorSemantic::visit_snapshot(std::shared_ptr<Snapshot> snapshot) {
	std::cout << "Registering snapshot " << snapshot->name.value() << std::endl;
	if (global.snapshots.find(snapshot->name) != global.snapshots.end()) {
		throw std::runtime_error(std::string(snapshot->begin()) + ": Error: snapshot with name " + snapshot->name.value() + 
			" already exists");
	}

	if (snapshot->parent_name) {
		auto found = global.snapshots.find(snapshot->parent_name);

		if (found == global.snapshots.end()) {
			throw std::runtime_error(std::string(snapshot->begin()) + ": Error: cannot find parent " + snapshot->parent_name.value() + 
				" for snapshot " + snapshot->name.value());
		} else {
			snapshot->parent = found->second;
		}
	}
	
	if (!global.snapshots.insert({snapshot->name, snapshot}).second) {
		throw std::runtime_error(std::string(snapshot->begin()) + ": Error while registering snapshot with name " +
			snapshot->name.value());
	}

	visit_action_block(snapshot->action_block->action); //dummy controller to match the interface
}

void VisitorSemantic::visit_test(std::shared_ptr<Test> test) {
	for (auto state: test->vms) {
		visit_vm_state(state);
	}
	visit_command_block(test->cmd_block);
	global.local_vms.clear();
}

void VisitorSemantic::visit_vm_state(std::shared_ptr<VmState> vm_state) {
	auto vm = global.vms.find(vm_state->name);
	if (vm == global.vms.end()) {
		throw std::runtime_error(std::string(vm_state->begin()) + ": Error: unknown vm name: " + vm_state->name.value());
	}

	if (vm_state->snapshot_name) {
		auto snapshot = global.snapshots.find(vm_state->snapshot_name);
		if (snapshot == global.snapshots.end()) {
			throw std::runtime_error(std::string(vm_state->begin()) + ": Error: unknown snapshot: " + vm_state->snapshot_name.value());
		}
		vm_state->snapshot = snapshot->second;
	}

	global.local_vms.insert({vm_state->name, vm->second});
}

void VisitorSemantic::visit_command_block(std::shared_ptr<CmdBlock> block) {
	for (auto command: block->commands) {
		visit_command(command);
	}
}


void VisitorSemantic::visit_command(std::shared_ptr<Cmd> cmd) {
	for (auto vm_token: cmd->vms) {
		auto vm = global.local_vms.find(vm_token.value());
		if (vm == global.local_vms.end()) {
			throw std::runtime_error(std::string(vm_token.pos()) + ": Error: unknown vm name: " + vm_token.value());
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
	} else if (auto p = std::dynamic_pointer_cast<Action<Set>>(action)) {
		return visit_set(p->action);
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
		if (global.fds.find(plug->name()) == global.fds.end()) {
			throw std::runtime_error(std::string(plug->begin()) + ": Error: Unknown flash drive: " + plug->name());
		}
	}
}

void VisitorSemantic::visit_exec(std::shared_ptr<Exec> exec) {
	if (exec->process_token.value() != "bash") {
		throw std::runtime_error(std::string(exec->begin()) + ": Error: unknown process name: " + exec->process_token.value());
	}
}

void VisitorSemantic::visit_set(std::shared_ptr<Set> set) {
	for (auto assign: set->assignments) {
		auto attr = assign->left.value();
		if ((attr != "login") &&
			(attr != "password")) {
			throw std::runtime_error(std::string(assign->begin()) + ": error: unknown attribute: " + attr);
		}
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
	if (global.vms.find(machine->name) != global.vms.end()) {
		throw std::runtime_error(std::string(machine->begin()) + ": Error: machine with name " + machine->name.value() + 
			" already exists");
	}

	auto config = visit_attr_block(machine->attr_block, "vm_global");
	config["name"] = machine->name.value();

	auto vm = std::shared_ptr<VmController>(new VmController(config));
	global.vms.emplace(std::make_pair(machine->name, vm));
}

void VisitorSemantic::visit_flash(std::shared_ptr<Controller> flash) {
	std::cout << "Registering flash " << flash->name.value() << std::endl;
	if (global.fds.find(flash->name) != global.fds.end()) {
		throw std::runtime_error(std::string(flash->begin()) + ": Error: flash drive with name " + flash->name.value() + 
			" already exists");
	}

	auto config = visit_attr_block(flash->attr_block, "fd_global");
	config["name"] = flash->name.value();

	auto fd = std::shared_ptr<FlashDriveController>(new FlashDriveController(config));
	global.fds.emplace(std::make_pair(flash->name, fd));
}

nlohmann::json VisitorSemantic::visit_attr_block(std::shared_ptr<AttrBlock> attr_block, const std::string& ctx_name) {
	nlohmann::json config;
	for (auto attr: attr_block->attrs) {
		visit_attr(attr, config, ctx_name);
	}
	return config;
}


void VisitorSemantic::visit_attr(std::shared_ptr<Attr> attr, nlohmann::json& config, const std::string& ctx_name) {
	if (ctx_name == "metadata") {
		if (value.type() != Token::category::dbl_quoted_string) {
			throw std::runtime_error(std::string(attr->begin()) + ": Error: metadata supports only double qouted strings: " + value.value()); 
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

	if (auto p = std::dynamic_pointer_cast<AttrValue<SimpleAttr>>(attr->value)) { 
		auto value = p->attr_value->t; 
		if (value.type() == Token::category::number) {
			config[attr->name.value()] = std::stoul(value.value()); 
		} else if (value.type() == Token::category::dbl_quoted_string) {
			config[attr->name.value()] = value.value().substr(1, value.value().length() - 2); //discard double qoutes 
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

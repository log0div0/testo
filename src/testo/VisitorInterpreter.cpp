
#include "VisitorInterpreter.hpp"

#include <fstream>
#include <thread>

using namespace AST;

static void sleep(const std::string& interval) {
	uint32_t seconds_to_sleep = std::stoul(interval.substr(0, interval.length() - 1));
	if (interval[interval.length() - 1] == 's') {
		seconds_to_sleep = seconds_to_sleep * 1;
	} else if (interval[interval.length() - 1] == 'm') {
		seconds_to_sleep = seconds_to_sleep * 60;
	} else if (interval[interval.length() - 1] == 'h') {
		seconds_to_sleep = seconds_to_sleep * 60 * 60;
	} else {
		throw std::runtime_error("Unknown time specifier"); //should not happen ever
	}

	std::this_thread::sleep_for(std::chrono::seconds(seconds_to_sleep));
}

void VisitorInterpreter::visit(std::shared_ptr<Program> program) {
	for (auto stmt: program->stmts) {
		visit_stmt(stmt);
	}
}

void VisitorInterpreter::visit_stmt(std::shared_ptr<IStmt> stmt) {
	if (auto p = std::dynamic_pointer_cast<Stmt<Test>>(stmt)) {
		return visit_test(p->stmt);
	} else if (auto p = std::dynamic_pointer_cast<Stmt<Controller>>(stmt)) {
		return visit_controller(p->stmt);
	}
}

void VisitorInterpreter::visit_controller(std::shared_ptr<Controller> controller) {
	if (controller->t.type() == Token::category::flash) {
		return visit_flash(controller);
	}
}

void VisitorInterpreter::visit_flash(std::shared_ptr<Controller> flash) {
	std::cout << "Creating flash drive " << flash->name.value() << std::endl;

	auto fd = global.fds.find(flash->name)->second; //should always be found

	if (fd->create()) {
		throw std::runtime_error(std::string(flash->begin()) + ": Error while creating flash drive " + flash->name.value());
	}

	if (fd->has_folder()) {
		std::cout << "Loading folder to flash drive " << fd->name() << std::endl;
		if (fd->load_folder()) {
			throw std::runtime_error(std::string(flash->begin()) + ": Error while loading folder to flash drive " +
				flash->name.value());
		}

	}
}

void VisitorInterpreter::visit_test(std::shared_ptr<Test> test) {
	std::cout << "Running test \"" << test->name.value() << "\"...\n";

	for (auto state: test->vms) {
		visit_vm_state(state);
	}

	visit_command_block(test->cmd_block);

	global.local_vms.clear();

	std::cout << "Test \"" << test->name.value() << "\" passed\n";
}

void VisitorInterpreter::visit_vm_state(std::shared_ptr<VmState> vm_state) {
	auto vm = global.vms.find(vm_state->name)->second;

	global.local_vms.insert({vm_state->name, vm});
	if (!vm_state->snapshot) {
		if (vm->install()) {
			throw std::runtime_error(std::string(vm_state->begin()) +
				": Error while performing install: " +
				std::string(*vm_state) +
				" on VM " +
				vm_state->name.value());
		}
		return;
	}

	if ((!vm->is_defined()) || (vm->get_metadata("vm_config_cksum") != vm->config_cksum())) {
		if (vm->install()) {
			throw std::runtime_error(std::string(vm_state->begin()) +
				": Error while performing install: " +
				std::string(*vm_state) +
				" on VM " +
				vm_state->name.value());
		}
		return apply_actions(vm, vm_state->snapshot, true);
	}

	if (resolve_state(vm, vm_state->snapshot)) {
		//everything is A-OK. We can rollback to the last snapshot
		if (vm->rollback(vm_state->snapshot->name)) {
			throw std::runtime_error(std::string(vm_state->snapshot->begin()) +
				": Error while performing rollback: " +
				vm_state->snapshot->name.value() +
				" on VM " +
				vm->name());
		}
	}
}

void VisitorInterpreter::visit_command_block(std::shared_ptr<CmdBlock> block) {
	for (auto command: block->commands) {
		visit_command(command);
	}
}

void VisitorInterpreter::visit_command(std::shared_ptr<Cmd> cmd) {
	for (auto vm_token: cmd->vms) {
		auto vm = global.local_vms.find(vm_token.value());
		visit_action(vm->second, cmd->action);
	}
}


void VisitorInterpreter::visit_action_block(std::shared_ptr<VmController> vm, std::shared_ptr<ActionBlock> action_block) {
	for (auto action: action_block->actions) {
		visit_action(vm, action);
	}
}

void VisitorInterpreter::visit_action(std::shared_ptr<VmController> vm, std::shared_ptr<IAction> action) {
	if (auto p = std::dynamic_pointer_cast<Action<Type>>(action)) {
		return visit_type(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Wait>>(action)) {
		return visit_wait(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Press>>(action)) {
		return visit_press(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Plug>>(action)) {
		return visit_plug(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Start>>(action)) {
		return visit_start(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Stop>>(action)) {
		return visit_stop(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Exec>>(action)) {
		return visit_exec(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Set>>(action)) {
		return visit_set(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<CopyTo>>(action)) {
		return visit_copyto(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<MacroCall>>(action)) {
		return visit_macro_call(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<IfClause>>(action)) {
		return visit_if_clause(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<ActionBlock>>(action)) {
		return visit_action_block(vm, p->action);
	} else {
		throw std::runtime_error("Unknown action");
	}
}

void VisitorInterpreter::visit_type(std::shared_ptr<VmController> vm, std::shared_ptr<Type> type) {
	std::string text = visit_word(vm, type->text_word);
	std::cout << "Typing " << text << " on vm " << vm->name() << std::endl;
	if (vm->type(text)) {
		throw std::runtime_error(std::string(type->begin()) +
			": Error while performing action: " +
			std::string(*type) +
			" on VM " +
			vm->name());
	}
}

void VisitorInterpreter::visit_wait(std::shared_ptr<VmController> vm, std::shared_ptr<Wait> wait) {
	std::string text = "";
	if (wait->text_word) {
		text = visit_word(vm, wait->text_word);
	}

	std::cout << "Waiting " << text << " on vm " << vm->name();
	if (wait->time_interval) {
		std::cout << " for " << wait->time_interval.value();
	}

	std::cout << std::endl;

	if (!wait->text_word) {
		return sleep(wait->time_interval.value());
	}

	std::string wait_for = wait->time_interval ? wait->time_interval.value() : "10s";

	if (vm->wait(text, wait_for)) {
		throw std::runtime_error(std::string(wait->begin()) +
			": Error while performing action: " +
			std::string(*wait) +
			" on VM " +
			vm->name());
	}
}

void VisitorInterpreter::visit_press(std::shared_ptr<VmController> vm, std::shared_ptr<Press> press) {
	for (auto key_spec: press->keys) {
		visit_key_spec(vm, key_spec);
	}
}

void VisitorInterpreter::visit_key_spec(std::shared_ptr<VmController> vm, std::shared_ptr<KeySpec> key_spec) {
	uint32_t times = key_spec->get_times();

	std::cout << "Pressing button " << key_spec->get_buttons_str();

	if (times > 1) {
		std::cout << " for " << times << " times ";
	}

	std::cout << " on vm " << vm->name() << std::endl;

	for (uint32_t i = 0; i < times; i++) {
		if (vm->press(key_spec->get_buttons())) {
			throw std::runtime_error(std::string(key_spec->begin()) +
			": Error while pressing buttons: " +
			std::string(*key_spec) +
			" on VM " +
			vm->name());
		}
	}
}

void VisitorInterpreter::visit_plug(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	if (plug->type.value() == "nic") {
		return visit_plug_nic(vm, plug);
	} else if (plug->type.value() == "link") {
		return visit_plug_link(vm, plug);
	} else if (plug->type.value() == "dvd") {
		return visit_plug_dvd(vm, plug);
	} else if (plug->type.value() == "flash") {
		if(plug->is_on()) {
			return plug_flash(vm, plug);
		} else {
			return unplug_flash(vm, plug);
		}
	} else {
		throw std::runtime_error(std::string(plug->begin()) + ":Error: unknown hardware type to plug/unplug: " +
			plug->type.value());
	}
}

void VisitorInterpreter::visit_plug_nic(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vm while semantic analisys
	auto nics = vm->nics();
	if (nics.find(plug->name()) == nics.end()) {
		throw std::runtime_error(std::string(plug->end()) + ": Error: unknown NIC " + plug->name() +
			" in VM " + vm->name());
	}

	std::string plug_unplug = plug->is_on() ? "plugging" : "unplugging";
	std::cout << plug_unplug << " nic " << plug->name() << " on vm " << vm->name() << std::endl;

	int result = 0;
	result = vm->set_nic(plug->name(), plug->is_on());

	if (result) {
		throw std::runtime_error(std::string(plug->begin()) + ": Error while " + plug_unplug +
			" nic on vm " + vm->name());
	}
}

void VisitorInterpreter::visit_plug_link(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vm while semantic analisys
	auto nics = vm->nics();
	if (nics.find(plug->name()) == nics.end()) {
		throw std::runtime_error(std::string(plug->end()) + ": Error: unknown NIC " + plug->name() +
			" in VM " + vm->name());
	}

	std::string plug_unplug = plug->is_on() ? "plugging" : "unplugging";
	std::cout << plug_unplug << " link " << plug->name() << " on vm " << vm->name() << std::endl;

	int result = 0;
	result = vm->set_link(plug->name(), plug->is_on());

	if (result) {
		throw std::runtime_error(std::string(plug->begin()) + ": Error while " + plug_unplug +
			" link on vm " + vm->name());
	}
}

void VisitorInterpreter::plug_flash(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	auto fd = global.fds.find(plug->name())->second; //should always be found
	std::cout << "Plugging flash drive " << fd->name() << " in vm " << vm->name() << std::endl;
	if (vm->is_plugged(fd)) {
		throw std::runtime_error(std::string(plug->begin()) + ": Error while plugging flash drive " + fd->name() +
			" in vm " + vm->name() + ": this flash drive is already plugged into " + fd->current_vm);
	}

	if (vm->plug_flash_drive(fd)) {
		throw std::runtime_error(std::string(plug->begin()) + ": Error while plugging flash drive " + fd->name() +
			" in vm " + vm->name());
	}
}

void VisitorInterpreter::unplug_flash(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	auto fd = global.fds.find(plug->name())->second; //should always be found
	std::cout << "Unplugging flash drive " << fd->name() << " from vm " << vm->name() << std::endl;
	if (!vm->is_plugged(fd)) {
		throw std::runtime_error(std::string(plug->begin()) + ": Error while unplugging flash drive " + fd->name() +
			" from vm " + vm->name() + ": this flash drive is not plugged to this vm");
	}

	if (vm->unplug_flash_drive(fd)) {
		throw std::runtime_error(std::string(plug->begin()) + ": Error while unplugging flash drive " + fd->name() +
				" from vm " + vm->name());
	}
}

void VisitorInterpreter::visit_plug_dvd(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	if (plug->is_on()) {
		std::cout << "Plugging dvd " << plug->name() << " in vm " << vm->name() << std::endl;
		if (vm->plug_dvd(plug->name())) {
			throw std::runtime_error(std::string(plug->begin()) + ": Error while plugging dvd " + plug->name() +
				" from vm " + vm->name());
		}
	} else {
		std::cout << "Unlugging dvd from vm " << vm->name() << std::endl;
		if (vm->unplug_dvd()) {
			throw std::runtime_error(std::string(plug->begin()) + ": Error while unplugging dvd from vm " + vm->name());
		}
	}
}

void VisitorInterpreter::visit_start(std::shared_ptr<VmController> vm, std::shared_ptr<Start> start) {
	std::cout << "Starting vm " << vm->name() << std::endl;
	if (vm->start()) {
		throw std::runtime_error(std::string(start->begin()) +
			": Error while performing start: " +
			std::string(*start) +
			" on VM " +
			vm->name());
	}
}

void VisitorInterpreter::visit_stop(std::shared_ptr<VmController> vm, std::shared_ptr<Stop> stop) {
	std::cout << "Stopping vm " << vm->name() << std::endl;
	if (vm->stop()) {
		throw std::runtime_error(std::string(stop->begin()) +
			": Error while performing stop: " +
			std::string(*stop) +
			" on VM " +
			vm->name());
	}
}

void VisitorInterpreter::visit_exec(std::shared_ptr<VmController> vm, std::shared_ptr<Exec> exec) {
	std::cout << "Executing  " << exec->process_token.value() << " command on vm " << vm->name() << std::endl;

	if (!vm->is_running()) {
		throw std::runtime_error(std::string(exec->begin()) + ": Error: vm " + vm->name() + " is not running");
	}

	if (!vm->is_additions_installed()) {
		throw std::runtime_error(std::string(exec->begin()) + ": Error: vbox additions are not installed on vm " + vm->name());
	}

	if (exec->process_token.value() == "bash") {
		//In future this should be a function

		std::string script = "set -e; set -o pipefail; set -x;";
		script += exec->script();

		//copy the script to tmp folder
		std::hash<std::string> h;

		std::string hash = std::to_string(h(script));

		fs::path host_script_dir = scripts_tmp_dir() / hash;
		fs::path guest_script_dir = fs::path("/tmp") / hash;

		if (!fs::create_directory(host_script_dir) && !fs::exists(host_script_dir)) {
			throw std::runtime_error(std::string(exec->begin()) + ": Error: can't create tmp script file on host");
		}

		fs::path host_script_file = host_script_dir / std::string(hash + ".sh");
		fs::path guest_script_file = guest_script_dir / std::string(hash + ".sh");
		std::ofstream script_stream(host_script_file);
		if (!script_stream.is_open()) {
			throw std::runtime_error(std::string(exec->begin()) + ": Error: Can't open tmp file for writing the script");
		}

		script_stream << script;
		script_stream.close();

		if (vm->copy_to_guest(host_script_dir, fs::path("/tmp"))) {
			throw std::runtime_error(std::string(exec->begin()) + ": Error: can't copy script file to vm");
		}

		fs::remove(host_script_file.generic_string());
		fs::remove(host_script_dir.generic_string());

		if (vm->run("/bin/bash", {guest_script_file.generic_string()})) {
			throw std::runtime_error(std::string(exec->begin()) + ": Error: one of the commands failed");
		}

		if (vm->remove_from_guest(guest_script_dir)) {
			throw std::runtime_error(std::string(exec->begin()) + ": Error: can't cleanup tmp file with script from guest");
		}
	}
}

void VisitorInterpreter::visit_set(std::shared_ptr<VmController> vm, std::shared_ptr<Set> set) {
	std::cout << "Setting attributes on vm " << vm->name() << std::endl;

	//TODO: redo!
	for (auto assign: set->assignments) {
		std::cout << assign->left.value() << " -> " << assign->value() << std::endl;
		vm->set_metadata(assign->left.value(), assign->value());
	}
}

void VisitorInterpreter::visit_copyto(std::shared_ptr<VmController> vm, std::shared_ptr<CopyTo> copyto) {
	std::cout << "Copying " << copyto->from() << " to vm " << vm->name() << " in directory " << copyto->to() << std::endl;

	if (!vm->is_running()) {
		throw std::runtime_error(std::string(copyto->begin()) + ": Error: vm " + vm->name() + " is not running");
	}

	if (!vm->is_additions_installed()) {
		throw std::runtime_error(std::string(copyto->begin()) + ": Error: vbox additions are not installed on vm " + vm->name());
	}

	if (vm->copy_to_guest(copyto->from(), copyto->to())) {
		throw std::runtime_error(std::string(copyto->begin()) + ": Error: copy to command failed");
	}
}

void VisitorInterpreter::visit_macro_call(std::shared_ptr<VmController> vm, std::shared_ptr<MacroCall> macro_call) {
	std::cout << "Calling macro " << macro_call->name().value() << " on vm " << vm->name() << std::endl;
	visit_action_block(vm, macro_call->macro->action_block->action);
}

void VisitorInterpreter::visit_if_clause(std::shared_ptr<VmController> vm, std::shared_ptr<IfClause> if_clause) {
	bool expr_result = visit_expr(vm, if_clause->expr);

	if (expr_result) {
		return visit_action(vm, if_clause->if_action);
	} else if (if_clause->has_else()) {
		return visit_action(vm, if_clause->else_action);
	}
}

bool VisitorInterpreter::visit_expr(std::shared_ptr<VmController> vm, std::shared_ptr<IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<Expr<BinOp>>(expr)) {
		return visit_binop(vm, p->expr);
	} else if (auto p = std::dynamic_pointer_cast<Expr<IFactor>>(expr)) {
		return visit_factor(vm, p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

bool VisitorInterpreter::visit_binop(std::shared_ptr<VmController> vm, std::shared_ptr<BinOp> binop) {
	auto left = visit_expr(vm, binop->left);
	auto right = visit_expr(vm, binop->right);

	if (binop->op().type() == Token::category::AND) {
		return left && right;
	} else if (binop->op().type() == Token::category::OR) {
		return left || right;
	} else {
		throw std::runtime_error("Unknown binop operation");
	}
}

bool VisitorInterpreter::visit_factor(std::shared_ptr<VmController> vm, std::shared_ptr<IFactor> factor) {
	if (auto p = std::dynamic_pointer_cast<Factor<Word>>(factor)) {
		return p->is_negated() ^ visit_word(vm, p->factor).length();
	} else if (auto p = std::dynamic_pointer_cast<Factor<Comparison>>(factor)) {
		return p->is_negated() ^ visit_comparison(vm, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<IExpr>>(factor)) {
		return p->is_negated() ^ visit_expr(vm, p->factor);
	} else {
		throw std::runtime_error("Unknown factor type");
	}
}

std::string VisitorInterpreter::resolve_var(std::shared_ptr<VmController> vm, const std::string& var) {
	//Resolving order
	//1) metadata
	//2) global (todo)
	//3) env var

	std::cout << "Resolving var " << var << std::endl;

	if (vm->has_key(var)) {
		return vm->get_metadata(var);
	}

	auto env_value = std::getenv(var.c_str());

	if (env_value == nullptr) {
		return "";
	}
	return env_value;
}

std::string VisitorInterpreter::visit_word(std::shared_ptr<VmController> vm, std::shared_ptr<Word> word) {
	std::string result;

	for (auto part: word->parts) {
		if (part.type() == Token::category::dbl_quoted_string) {
			result += part.value().substr(1, part.value().length() - 2);
		} else if (part.type() == Token::category::var_ref) {
			result += resolve_var(vm, part.value().substr(1, part.value().length() - 1));
		} else if (part.type() == Token::category::multiline_string) {
			result += part.value().substr(3, part.value().length() - 6);
		} else {
			throw std::runtime_error("Unknown word type");
		}
	}

	return result;
}

bool VisitorInterpreter::visit_comparison(std::shared_ptr<VmController> vm, std::shared_ptr<Comparison> comparison) {
	auto left = visit_word(vm, comparison->left);
	auto right = visit_word(vm, comparison->right);
	if (comparison->op() == Token::category::GREATER) {
		if (!is_number(left)) {
			throw std::runtime_error(std::string(comparison->left->begin()) + ": Error: " + std::string(*comparison->left) + " is not an integer number");
		}
		if (!is_number(right)) {
			throw std::runtime_error(std::string(comparison->right->begin()) + ": Error: " + std::string(*comparison->right) + " is not an integer number");
		}

		return std::stoul(left) > std::stoul(right);

	} else if (comparison->op() == Token::category::LESS) {
		if (!is_number(left)) {
			throw std::runtime_error(std::string(comparison->left->begin()) + ": Error: " + std::string(*comparison->left) + " is not an integer number");
		}
		if (!is_number(right)) {
			throw std::runtime_error(std::string(comparison->right->begin()) + ": Error: " + std::string(*comparison->right) + " is not an integer number");
		}

		return std::stoul(left) < std::stoul(right);

	} else if (comparison->op() == Token::category::EQUAL) {
		if (!is_number(left)) {
			throw std::runtime_error(std::string(comparison->left->begin()) + ": Error: " + std::string(*comparison->left) + " is not an integer number");
		}
		if (!is_number(right)) {
			throw std::runtime_error(std::string(comparison->right->begin()) + ": Error: " + std::string(*comparison->right) + " is not an integer number");
		}

		return std::stoul(left) == std::stoul(right);

	} else if (comparison->op() == Token::category::STRGREATER) {
		return left > right;
	} else if (comparison->op() == Token::category::STRLESS) {
		return left < right;
	} else if (comparison->op() == Token::category::STREQUAL) {
		return left == right;
	} else {
		throw std::runtime_error("Unknown comparison op");
	}
}

void VisitorInterpreter::apply_actions(std::shared_ptr<VmController> vm, std::shared_ptr<Snapshot> snapshot, bool recursive) {
	if (recursive) {
		if (snapshot->parent) {
			apply_actions(vm, snapshot->parent, true);
		}
	}

	std::cout << "Applying snapshot " << snapshot->name.value() << " to vm " << vm->name() << std::endl;
	visit_action_block(vm, snapshot->action_block->action);
	auto new_cksum = cksum(snapshot);
	std::cout << "Taking snapshot " << snapshot->name.value() << " for vm " << vm->name() << std::endl;
	if (vm->make_snapshot(snapshot->name)) {
		throw std::runtime_error(std::string(snapshot->begin()) + ": Error: error while creating snapshot" +
			snapshot->name.value() + " for vm " + vm->name());
	}
	std::cout << "Setting snapshot " << snapshot->name.value() << " cksum: " << new_cksum << std::endl;
	if (vm->set_snapshot_cksum(snapshot->name, new_cksum)) {
		throw std::runtime_error(std::string(snapshot->begin()) + ": Error: error while setting snapshot cksum" +
			snapshot->name.value() + " for vm " + vm->name());
	}
}

//return true if everything is OK and we don't need to apply nothing
bool VisitorInterpreter::resolve_state(std::shared_ptr<VmController> vm, std::shared_ptr<Snapshot> snapshot) {
	bool parents_are_ok = true;

	if (snapshot->parent) {
		parents_are_ok = resolve_state(vm, snapshot->parent);
	}

	if (parents_are_ok) {
		if (vm->has_snapshot(snapshot->name)) {
			if (vm->get_snapshot_cksum(snapshot->name) == cksum(snapshot)) {
				return true;
			}
		}

		if (snapshot->parent) {
			if (vm->rollback(snapshot->parent->name)) {
				throw std::runtime_error(std::string(snapshot->parent->begin()) +
					": Error while performing rollback: " +
					snapshot->parent->name.value() +
					" on VM " +
					vm->name());
			}
		} else {
			if (vm->install()) {
				throw std::runtime_error(std::string(snapshot->begin()) +
					": Error while performing install: " +
					" on VM " +
					vm->name());
			}
		}
	}

	apply_actions(vm, snapshot);
	return false;
}


std::string VisitorInterpreter::cksum(std::shared_ptr<Snapshot> snapshot) {
	std::string combined = std::string(*snapshot);

	for (auto it = snapshot->parent; it != nullptr; it = it->parent) {
		combined += std::string(*it);
	}
	std::hash<std::string> h;

	auto result = h(combined);
	return std::to_string(result);
}

std::string VisitorInterpreter::cksum(std::shared_ptr<Controller> flash) {
	std::hash<std::string> h;
	auto result = h(std::string(*flash));

	return std::to_string(result);
}

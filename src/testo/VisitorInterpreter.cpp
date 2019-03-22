
#include "VisitorInterpreter.hpp"
#include "VisitorCksum.hpp"

#include <fmt/format.h>
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

void VisitorInterpreter::setup_progress_vars(std::shared_ptr<Program> program) {
	for (auto stmt: program->stmts) {
		if (auto p = std::dynamic_pointer_cast<Stmt<Test>>(stmt)) {
			tests_num++;
		}
	}

	if (tests_num != 0) {
		progress_step = 100 / tests_num;
		original_remainder = 100 % tests_num;
		current_remainder = original_remainder;
	} else {
		progress_step = 100;
	}

}

void VisitorInterpreter::update_progress() {
	current_progress += progress_step;
	if (original_remainder != 0) {
		if ((current_remainder / tests_num) > 0) {
			current_remainder = current_remainder / tests_num;
			current_progress++;
		}
		current_remainder += original_remainder;
	}
}

void VisitorInterpreter::visit(std::shared_ptr<Program> program) {
	try {
		setup_progress_vars(program);

		for (auto stmt: program->stmts) {
			visit_stmt(stmt);
		}
	}
	catch (const std::exception& error) {
		std::cout << error << std::endl;
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
	try {
		print("Creating flash drive \"", flash->name.value());

		auto fd = reg.fds.find(flash->name)->second; //should always be found

		fd->create();

		if (fd->has_folder()) {
			print("Loading folder to flash drive \"", fd->name());
			fd->load_folder();
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(flash, nullptr));
	}

}

void VisitorInterpreter::visit_test(std::shared_ptr<Test> test) {
	try {
		print("Running test \"", test->name.value(), "\"...");

		for (auto state: test->vms) {
			visit_vm_state(state);
		}

		//Let's remember all the keys all vms have
		std::unordered_map<std::shared_ptr<VmController>, std::vector<std::string>> original_states;

		for (auto state: test->vms) {
			auto vm = reg.vms.find(state->name)->second;
			auto keys = vm->keys();
			original_states.insert({vm, keys});
		}

		visit_command_block(test->cmd_block);

		for (auto state: original_states) {
			auto vm = state.first;
			auto original_keys = state.second;
			auto final_keys = vm->keys();
			std::sort(original_keys.begin(), original_keys.end());
			std::sort(final_keys.begin(), final_keys.end());
			std::vector<std::string> new_keys;
			std::set_difference(final_keys.begin(), final_keys.end(), original_keys.begin(), original_keys.end(), std::back_inserter(new_keys));
			for (auto& key: new_keys) {
				vm->set_metadata(key, "");
			}

			if (vm->is_flash_plugged(nullptr)) {
				throw std::runtime_error(fmt::format("Vm {} has unplugged flash drive, you must unplug it before the end of the test", vm->name()));
			}
		}

		reg.local_vms.clear();

		update_progress();
		print("Test \"", test->name.value(), "\" passed");
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
	}

}

void VisitorInterpreter::visit_vm_state(std::shared_ptr<VmState> vm_state) {
	try {
		auto vm = reg.vms.find(vm_state->name)->second;

		reg.local_vms.insert({vm_state->name, vm});

		if (!vm_state->snapshot) {
			vm->install();
			return;
		}

		if ((!vm->is_defined()) || !check_config_relevance(vm->get_config(), nlohmann::json::parse(vm->get_metadata("vm_config")))) {
			vm->install();
			return apply_actions(vm, vm_state->snapshot, true);
		}

		if (resolve_state(vm, vm_state->snapshot)) {
			//everything is A-OK. We can rollback to the last snapshot
			vm->rollback(vm_state->snapshot->name);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(vm_state, nullptr));
	}
}

void VisitorInterpreter::visit_command_block(std::shared_ptr<CmdBlock> block) {
	for (auto command: block->commands) {
		visit_command(command);
	}
}

void VisitorInterpreter::visit_command(std::shared_ptr<Cmd> cmd) {
	for (auto vm_token: cmd->vms) {
		auto vm = reg.local_vms.find(vm_token.value());
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
	} else if (auto p = std::dynamic_pointer_cast<Action<Empty>>(action)) {
		return;
	} else {
		throw std::runtime_error("Unknown action");
	}
}

void VisitorInterpreter::visit_type(std::shared_ptr<VmController> vm, std::shared_ptr<Type> type) {
	try {
		std::string text = visit_word(vm, type->text_word);
		print("Typing ", text, " on vm ", vm->name());
		vm->type(text);
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(type, vm));
	}
}

void VisitorInterpreter::visit_wait(std::shared_ptr<VmController> vm, std::shared_ptr<Wait> wait) {
	try {
		std::string text = "";
		if (wait->text_word) {
			text = visit_word(vm, wait->text_word);
		}

		std::string print_str = std::string("Waiting ") + text + " on vm " + vm->name();
		if (wait->time_interval) {
			print_str += " for " + wait->time_interval.value();
		}

		print(print_str);

		if (!wait->text_word) {
			return sleep(wait->time_interval.value());
		}

		std::string wait_for = wait->time_interval ? wait->time_interval.value() : "10s";
		if (!vm->wait(text, wait_for)) {
			throw std::runtime_error("Wait timeout");
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(wait, vm));
	}

}

void VisitorInterpreter::visit_press(std::shared_ptr<VmController> vm, std::shared_ptr<Press> press) {
	try {
		for (auto key_spec: press->keys) {
			visit_key_spec(vm, key_spec);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(press, vm));
	}
}

void VisitorInterpreter::visit_key_spec(std::shared_ptr<VmController> vm, std::shared_ptr<KeySpec> key_spec) {
	uint32_t times = key_spec->get_times();

	std::string print_str = std::string("Pressing button ") + key_spec->get_buttons_str();

	if (times > 1) {
		print_str += std::string(" ") + std::to_string(times) + " times ";
	}

	print_str += std::string(" on vm ") + vm->name();

	for (uint32_t i = 0; i < times; i++) {
		vm->press(key_spec->get_buttons());
	}
}

void VisitorInterpreter::visit_plug(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	try {
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
			throw std::runtime_error(std::string("unknown hardware type to plug/unplug: ") +
				plug->type.value());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(plug, vm));
	}
}

void VisitorInterpreter::visit_plug_nic(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vm while semantic analisys
	auto nic = plug->name_token.value();
	auto nics = vm->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("specified nic {} is not present in this vm", nic));
	}

	if (vm->is_running()) {
		throw std::runtime_error(fmt::format("vm is running, but must be stopeed"));
	}

	if (vm->is_nic_plugged(nic) == plug->is_on()) {
		if (plug->is_on()) {
			throw std::runtime_error(fmt::format("specified nic {} is already plugged in this vm", nic));
		} else {
			throw std::runtime_error(fmt::format("specified nic {} is not unplugged from this vm", nic));
		}
	}

	std::string plug_unplug = plug->is_on() ? "plugging" : "unplugging";
	print(plug_unplug, " nic ", nic, " on vm ", vm->name());

	vm->set_nic(nic, plug->is_on());
}

void VisitorInterpreter::visit_plug_link(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vm while semantic analisys

	auto nic = plug->name_token.value();
	auto nics = vm->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("the nic for specified link {} is not present in this vm", nic));
	}

	if (!vm->is_nic_plugged(nic)) {
		throw std::runtime_error(fmt::format("the nic for specified link {} is unplugged, you must to plug it first", nic));
	}

	if (plug->is_on() == vm->is_link_plugged(nic)) {
		if (plug->is_on()) {
			throw std::runtime_error(fmt::format("specified link {} is already plugged in this vm", nic));
		} else {
			throw std::runtime_error(fmt::format("specified link {} is already unplugged from this vm", nic));
		}
	}

	std::string plug_unplug = plug->is_on() ? "plugging" : "unplugging";
	print(plug_unplug, " link ", nic, " on vm ", vm->name());

	vm->set_link(nic, plug->is_on());
}

void VisitorInterpreter::plug_flash(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	auto fd = reg.fds.find(plug->name_token.value())->second; //should always be found
	print("Plugging flash drive ", fd->name(), " in vm ", vm->name());
	if (vm->is_flash_plugged(fd)) {
		throw std::runtime_error(fmt::format("specified flash {} is already plugged into this vm", fd->name()));
	}

	vm->plug_flash_drive(fd);
}

void VisitorInterpreter::unplug_flash(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	auto fd = reg.fds.find(plug->name_token.value())->second; //should always be found
	print("Unlugging flash drive ", fd->name(), " from vm ", vm->name());
	if (!vm->is_flash_plugged(fd)) {
		throw std::runtime_error(fmt::format("specified flash {} is already unplugged from this vm", fd->name()));
	}

	vm->unplug_flash_drive(fd);
}

void VisitorInterpreter::visit_plug_dvd(std::shared_ptr<VmController> vm, std::shared_ptr<Plug> plug) {
	if (plug->is_on()) {
		if (vm->is_dvd_plugged()) {
			throw std::runtime_error(fmt::format("some dvd is already plugged"));
		}

		auto path = visit_word(vm, plug->path);
		print("Plugging dvd ", path, " in vm ", vm->name());
		vm->plug_dvd(path);
	} else {
		if (!vm->is_dvd_plugged()) {
			throw std::runtime_error(fmt::format("dvd is already unplugged"));
		}

		print("Plugging dvd from vm ", vm->name());
		vm->unplug_dvd();
	}
}

void VisitorInterpreter::visit_start(std::shared_ptr<VmController> vm, std::shared_ptr<Start> start) {
	try {
		print("Starting vm ", vm->name());
		vm->start();
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(start, vm));
	}
}

void VisitorInterpreter::visit_stop(std::shared_ptr<VmController> vm, std::shared_ptr<Stop> stop) {
	try {
		print("Stopping vm ", vm->name());
		vm->stop();
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(stop, vm));

	}

}

void VisitorInterpreter::visit_exec(std::shared_ptr<VmController> vm, std::shared_ptr<Exec> exec) {
	try {
		print("Executing ", exec->process_token.value(), " command on vm ", vm->name());

		if (!vm->is_running()) {
			throw std::runtime_error(fmt::format("vm is not running"));
		}

		if (!vm->is_additions_installed()) {
			throw std::runtime_error(fmt::format("guest additions is not installed"));
		}

		if (exec->process_token.value() == "bash") {
			//In future this should be a function

			std::string script = "set -e; set -o pipefail; set -x;";
			script += visit_word(vm, exec->commands);

			//copy the script to tmp folder
			std::hash<std::string> h;

			std::string hash = std::to_string(h(script));

			fs::path host_script_dir = scripts_tmp_dir() / hash;
			fs::path guest_script_dir = fs::path("/tmp");

			if (!fs::create_directories(host_script_dir) && !fs::exists(host_script_dir)) {
				throw std::runtime_error(fmt::format("can't create tmp script file on host"));
			}

			fs::path host_script_file = host_script_dir / std::string(hash + ".sh");
			fs::path guest_script_file = guest_script_dir / std::string(hash + ".sh");
			std::ofstream script_stream(host_script_file);
			if (!script_stream.is_open()) {
				throw std::runtime_error(fmt::format("Can't open tmp file for writing the script"));
			}

			script_stream << script;
			script_stream.close();

			vm->copy_to_guest(host_script_dir, fs::path("/tmp"));

			fs::remove(host_script_file.generic_string());
			fs::remove(host_script_dir.generic_string());

			vm->run("/bin/bash", {guest_script_file.generic_string()});
			vm->remove_from_guest(guest_script_dir);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(exec, vm));
	}
}

void VisitorInterpreter::visit_set(std::shared_ptr<VmController> vm, std::shared_ptr<Set> set) {
	try {
		print("Setting attributes on vm ", vm->name());

		//1) Let's check all the assignments so that we know we don't override any values

		for (auto assign: set->assignments) {
			if (vm->has_key(assign->left.value())) {
				throw std::runtime_error(fmt::format("Can't override key {}", assign->left.value()));
			}
		}

		for (auto assign: set->assignments) {
			std::string value = visit_word(vm, assign->right);
			std::cout << assign->left.value() << " -> " << value << std::endl;
			vm->set_metadata(assign->left.value(), value);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(set, vm));
	}

}

void VisitorInterpreter::visit_copyto(std::shared_ptr<VmController> vm, std::shared_ptr<CopyTo> copyto) {
	try {
		auto from = visit_word(vm, copyto->from);
		auto to = visit_word(vm, copyto->to);

		print("Copying ", from, " to vm ", vm->name(), " in directory ", to);

		if (!vm->is_running()) {
			throw std::runtime_error(fmt::format("vm is not running"));
		}

		if (!vm->is_additions_installed()) {
			throw std::runtime_error(fmt::format("guest additions are not installed"));
		}

		vm->copy_to_guest(from, to);
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(copyto, vm));
	}

}

void VisitorInterpreter::visit_macro_call(std::shared_ptr<VmController> vm, std::shared_ptr<MacroCall> macro_call) {
	print("Calling macro ", macro_call->name().value(), " on vm ", vm->name());
	//push new ctx
	Stack new_ctx;

	for (size_t i = 0; i < macro_call->params.size(); ++i) {
		auto value = visit_word(vm, macro_call->params[i]);
		new_ctx.define(macro_call->macro->params[i].value(), value);
	}

	local_vars.push(new_ctx);
	visit_action_block(vm, macro_call->macro->action_block->action);
	//pop ctx
	local_vars.pop();
}

void VisitorInterpreter::visit_if_clause(std::shared_ptr<VmController> vm, std::shared_ptr<IfClause> if_clause) {
	try {
		bool expr_result = visit_expr(vm, if_clause->expr);

		if (expr_result) {
			return visit_action(vm, if_clause->if_action);
		} else if (if_clause->has_else()) {
			return visit_action(vm, if_clause->else_action);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(InterpreterException(if_clause, vm));
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
	//2) reg (todo)
	//3) env var

	print("Resolving var ", var);

	if (!local_vars.empty()) {
		auto top = local_vars.top();
		if (top.is_defined(var)) {
			return top.ref(var);
		}
	}

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
			throw std::runtime_error(std::string(*comparison->left) + " is not an integer number");
		}
		if (!is_number(right)) {
			throw std::runtime_error(std::string(*comparison->right) + " is not an integer number");
		}

		return std::stoul(left) > std::stoul(right);

	} else if (comparison->op() == Token::category::LESS) {
		if (!is_number(left)) {
			throw std::runtime_error(std::string(*comparison->left) + " is not an integer number");
		}
		if (!is_number(right)) {
			throw std::runtime_error(std::string(*comparison->right) + " is not an integer number");
		}

		return std::stoul(left) < std::stoul(right);

	} else if (comparison->op() == Token::category::EQUAL) {
		if (!is_number(left)) {
			throw std::runtime_error(std::string(*comparison->left) + " is not an integer number");
		}
		if (!is_number(right)) {
			throw std::runtime_error(std::string(*comparison->right) + " is not an integer number");
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

	print("Applying snapshot ", snapshot->name.value(), " to vm ", vm->name());
	visit_action_block(vm, snapshot->action_block->action);

	if (vm->is_flash_plugged(nullptr)) {
		throw std::runtime_error(fmt::format("Can't take snapshot {}: you must unplug all flash drives", snapshot->name.value()));
	}

	auto new_cksum = snapshot_cksum(vm, snapshot);
	print("Taking snapshot ", snapshot->name.value(), " for vm ", vm->name());
	vm->make_snapshot(snapshot->name, new_cksum);
}

//return true if everything is OK and we don't need to apply nothing
bool VisitorInterpreter::resolve_state(std::shared_ptr<VmController> vm, std::shared_ptr<Snapshot> snapshot) {
	bool parents_are_ok = true;

	if (snapshot->parent) {
		parents_are_ok = resolve_state(vm, snapshot->parent);
	}

	if (parents_are_ok) {
		if (vm->has_snapshot(snapshot->name)) {
			if (vm->get_snapshot_cksum(snapshot->name) == snapshot_cksum(vm, snapshot)) {
				return true;
			}
		}

		if (snapshot->parent) {
			vm->rollback(snapshot->parent->name);
		} else {
			vm->install();
		}
	}

	apply_actions(vm, snapshot);
	return false;
}

bool VisitorInterpreter::check_config_relevance(nlohmann::json new_config, nlohmann::json old_config) const {
	//So....
	//1) get rid of metadata
	new_config.erase("metadata");
	old_config.erase("metadata");


	//2) Actually.... Let's just be practical here.
	//Check if both have or don't have nics

	auto old_nics = old_config.value("nic", nlohmann::json::array());
	auto new_nics = new_config.value("nic", nlohmann::json::array());

	if (old_nics.size() != new_nics.size()) {
		return false;
	}

	if (!std::is_permutation(old_nics.begin(), old_nics.end(), new_nics.begin())) {
		return false;
	}

	new_config.erase("nic");
	old_config.erase("nic");

	return (old_config == new_config);
}

std::string VisitorInterpreter::snapshot_cksum(std::shared_ptr<VmController> vm, std::shared_ptr<Snapshot> snapshot) {
	VisitorCksum visitor(reg);
	return std::to_string(visitor.visit(vm, snapshot));
}

std::string VisitorInterpreter::cksum(std::shared_ptr<Controller> flash) {
	std::hash<std::string> h;
	auto result = h(std::string(*flash));

	return std::to_string(result);
}

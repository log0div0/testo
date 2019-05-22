
#include "VisitorInterpreter.hpp"
#include "VisitorCksum.hpp"

#include "coro/Finally.h"
#include <fmt/format.h>
#include <fstream>
#include <thread>

using namespace AST;

template <typename Duration>
std::string duration_to_str(Duration duration) {

	auto h = std::chrono::duration_cast<std::chrono::hours>(duration);
	duration -= h;
	auto m = std::chrono::duration_cast<std::chrono::minutes>(duration);
	duration -= m;
	auto s = std::chrono::duration_cast<std::chrono::seconds>(duration);
	auto result = fmt::format("{}h:{}m:{}s", h.count(), m.count(), s.count());

	return result;
}

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

VisitorInterpreter::VisitorInterpreter(Register& reg, const nlohmann::json& config): reg(reg) {
	stop_on_fail = config.at("stop_on_fail").get<bool>();
	test_spec = config.at("test_spec").get<std::string>();
}

void VisitorInterpreter::print_statistics() const {
	auto total_tests = succeeded_tests.size() + failed_tests.size() + up_to_date_tests.size();
	auto tests_durantion = std::chrono::system_clock::now() - start_timestamp;

	std::cout << "PROCESSED TOTAL " << total_tests << " TESTS IN " << duration_to_str(tests_durantion) << std::endl;
	std::cout << "UP TO DATE: " << up_to_date_tests.size() << std::endl;
	std::cout << "RUN SUCCESSFULLY: " << succeeded_tests.size() << std::endl;
	std::cout << "FAILED: " << failed_tests.size() << std::endl;
	for (auto fail: failed_tests) {
		std::cout << "\t -" << fail->name.value() << std::endl;
	}
}

void VisitorInterpreter::setup_vars(std::shared_ptr<Program> program) {
	//Need to check that we don't have duplicates
	//And we can't use std::set because we need to
	//keep the order of the tests
	for (auto stmt: program->stmts) {
		if (auto p = std::dynamic_pointer_cast<Stmt<Test>>(stmt)) {
			auto test = p->stmt;

			//So for every test
			//we need to check if it's suitable for test spec
			//if it is - push back to list and remove all the parents duplicates

			if (test_spec.length() && (test->name.value() != test_spec)) {
				continue;
			}

			concat_unique(tests_to_run, reg.get_test_path(test));
		} else if (auto p = std::dynamic_pointer_cast<Stmt<Controller>>(stmt)) {
			if (p->stmt->t.type() == Token::category::flash) {
				flash_drives.push_back(p->stmt);
			}
		}
	}

	/*auto tests_num = tests_to_run.size();
	if (tests_num != 0) {
		progress_step = 100 / tests_num;
		original_remainder = 100 % tests_num;
		current_remainder = original_remainder;
	} else {
		progress_step = 100;
	}
*/
}

void VisitorInterpreter::update_progress() {
	/*current_progress += progress_step;
	if (original_remainder != 0) {
		if ((current_remainder / tests_to_run.size()) > 0) {
			current_remainder = current_remainder / tests_to_run.size();
			current_progress++;
		}
		current_remainder += original_remainder;
	}*/
}

void VisitorInterpreter::visit(std::shared_ptr<Program> program) {
	start_timestamp = std::chrono::system_clock::now();

	setup_vars(program);

	//Create flash drives
	for (auto fd: flash_drives) {
		visit_flash(fd);
	}

	if (tests_to_run.size() == 0) {
		if (test_spec.length()) {
			std::cout << "Couldn't find a test with the name " << test_spec << std::endl;
		} else {
			std::cout << "There's no tests to run\n";
		}
		return;
	}

	for (auto test: tests_to_run) {
		std::cout<< "Running test " << test->name.value() << std::endl;
		visit_test(test);
	}

	print_statistics();
}

void VisitorInterpreter::visit_controller(std::shared_ptr<Controller> controller) {
	if (controller->t.type() == Token::category::flash) {
		return visit_flash(controller);
	}
}

void VisitorInterpreter::visit_flash(std::shared_ptr<Controller> flash) {
	try {
		auto fd = reg.fds.find(flash->name)->second; //should always be found
		if (!fd->cache_enabled() || (cksum(fd) != fd->cksum())) {
			print("Creating flash drive \"", flash->name.value());
			fd->create();
			if (fd->has_folder()) {
				print("Loading folder to flash drive \"", fd->name());
				fd->load_folder();
			}
		} else {
			print("Using cached flash drive \"", flash->name.value());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(flash, nullptr));
	}

}

void VisitorInterpreter::visit_test(std::shared_ptr<Test> test) {
	try {
		//Check if one of the parents failed. If it did, just fail
		for (auto parent: test->parents) {
			for (auto failed: failed_tests) {
				if (parent == failed) {
					print("Skipping test ", test->name.value(), " because his parent ", parent->name.value(), " failed");
					failed_tests.push_back(test);
					return;
				}
			}
		}

		//Now check if out test is cached
		//Our test is checked if every vm in our test (parents included)
		// - is defined
		// - has valid config and dvd cksum
		// - has snapshots with corresponding name and valid cksums

		bool is_cached = true;

		for (auto vm: reg.get_all_vms(test)) {
			if (vm->is_defined() &&
				check_config_relevance(vm->get_config(), nlohmann::json::parse(vm->get_metadata("vm_config"))) &&
				(file_signature(vm->get_config().at("iso").get<std::string>()) == vm->get_metadata("dvd_signature")) &&
				vm->has_snapshot(test->name.value()) &&
				(vm->get_snapshot_cksum(test->name.value()) == test_cksum(test)))
			{
				continue;
			}
			is_cached = false;
		}

		if (is_cached) {
			print("Test ", test->name.value(), " is up-to-date, skipping...");
			up_to_date_tests.push_back(test);
			return;
		}

		//Ok, we're not cached and we need to run the test
		//First we need to get all the vms in the correct state
		//vms from parents - rollback to parent snapshot (we can be sure it is present and have valid cksum)
		//new vms - install

		for (auto parent: test->parents) {
			for (auto vm: reg.get_all_vms(parent)) {
				print("Restoring snapshot ", parent->name.value(), " for vm ", vm->name());
				vm->rollback(parent->name.value());
			}
		}

		for (auto vm: reg.get_all_vms(test)) {
			//check if it's a new one
			auto is_new = true;
			for (auto parent: test->parents) {
				auto parent_vms = reg.get_all_vms(parent);
				if (parent_vms.find(vm) != parent_vms.end()) {
					//not new, go to the next vm
					is_new = false;
					break;
				}
			}

			if (is_new) {
				print("Creating machine ", vm->name());
				vm->install();
			}
		}

		//Everything is in the right state so we could actually do the test
		visit_command_block(test->cmd_block);

		//But that's not everything - we need to create according snapshots to all included vms

		//TODO: pause all vms

		for (auto vm: reg.get_all_vms(test)) {
			print("Taking snapshot ", test->name.value(), " for vm ", vm->name());
			vm->make_snapshot(test->name.value(), test_cksum(test));
		}

		stop_all_vms(test);

		succeeded_tests.push_back(test);

	} catch (const InterpreterException& error) {
		std::cout << error << std::endl;
		print ("Test ", test->name.value(), " FAILED");
		failed_tests.push_back(test);

		if (stop_on_fail) {
			throw std::runtime_error("");
		}

		stop_all_vms(test);
	} //everything else is fatal and should be catched furter up
}

void VisitorInterpreter::visit_command_block(std::shared_ptr<CmdBlock> block) {
	for (auto command: block->commands) {
		visit_command(command);
	}
}

void VisitorInterpreter::visit_command(std::shared_ptr<Cmd> cmd) {
	for (auto vm_token: cmd->vms) {
		auto vm = reg.vms.find(vm_token.value());
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
	} else if (auto p = std::dynamic_pointer_cast<Action<Shutdown>>(action)) {
		return visit_shutdown(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Exec>>(action)) {
		return visit_exec(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Set>>(action)) {
		return visit_set(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Copy>>(action)) {
		return visit_copy(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<MacroCall>>(action)) {
		return visit_macro_call(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<IfClause>>(action)) {
		return visit_if_clause(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<ForClause>>(action)) {
		return visit_for_clause(vm, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<CycleControl>>(action)) {
		throw CycleControlException(p->action->t);
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
		std::throw_with_nested(ActionException(type, vm));
	}
}

void VisitorInterpreter::visit_wait(std::shared_ptr<VmController> vm, std::shared_ptr<Wait> wait) {
	try {
		std::string text = "";
		if (wait->text_word) {
			text = visit_word(vm, wait->text_word);
		}

		std::string print_str = std::string("Waiting ");
		if (text.length()) {
			print_str += "\"" + text + "\"";
		}
		nlohmann::json params = {};

		for (auto it = wait->params.begin(); it != wait->params.end();) {
			if (it == wait->params.begin()) {
				print_str += "(";
			}

			auto value = visit_word(vm, (*it)->right);
			params[(*it)->left.value()] = value;
			print_str += (*it)->left.value() + "=" + value;

			if (++it != wait->params.end()) {
				print_str += ", ";
				continue;
			}
			print_str += ")";
		}

		print_str += std::string(" on vm ") + vm->name();
		if (wait->time_interval) {
			print_str += " for " + wait->time_interval.value();
		}

		print(print_str);

		if (!wait->text_word) {
			return sleep(wait->time_interval.value());
		}

		std::string wait_for = wait->time_interval ? wait->time_interval.value() : "1m";
		if (!vm->wait(text, params, wait_for)) {
			throw std::runtime_error("Wait timeout");
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(wait, vm));
	}

}

void VisitorInterpreter::visit_press(std::shared_ptr<VmController> vm, std::shared_ptr<Press> press) {
	try {
		for (auto key_spec: press->keys) {
			visit_key_spec(vm, key_spec);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(press, vm));
	}
}

void VisitorInterpreter::visit_key_spec(std::shared_ptr<VmController> vm, std::shared_ptr<KeySpec> key_spec) {
	uint32_t times = key_spec->get_times();

	std::string print_str = std::string("Pressing button ") + key_spec->get_buttons_str();

	if (times > 1) {
		print_str += std::string(" ") + std::to_string(times) + " times ";
	}

	print_str += std::string(" on vm ") + vm->name();

	print(print_str);

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
		std::throw_with_nested(ActionException(plug, vm));
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

		fs::path path = visit_word(vm, plug->path);
		print("Plugging dvd ", path, " in vm ", vm->name());
		if (path.is_relative()) {
			path = plug->t.pos().file.parent_path() / path;
		}
		vm->plug_dvd(path);
	} else {
		if (!vm->is_dvd_plugged()) {
			throw std::runtime_error(fmt::format("dvd is already unplugged"));
		}

		print("Unplugging dvd from vm ", vm->name());
		vm->unplug_dvd();
	}
}

void VisitorInterpreter::visit_start(std::shared_ptr<VmController> vm, std::shared_ptr<Start> start) {
	try {
		print("Starting vm ", vm->name());
		vm->start();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(start, vm));
	}
}

void VisitorInterpreter::visit_stop(std::shared_ptr<VmController> vm, std::shared_ptr<Stop> stop) {
	try {
		print("Stopping vm ", vm->name());
		vm->stop();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(stop, vm));

	}
}

void VisitorInterpreter::visit_shutdown(std::shared_ptr<VmController> vm, std::shared_ptr<Shutdown> shutdown) {
	try {
		print("Shutting down vm ", vm->name());
		std::string wait_for = shutdown->time_interval ? shutdown->time_interval.value() : "1m";
		vm->shutdown(time_to_seconds(wait_for));
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(shutdown, vm));

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

			vm->copy_to_guest(host_script_dir, fs::path("/tmp"), 5); //5 seconds should be enough to pass any script

			fs::remove(host_script_file.generic_string());
			fs::remove(host_script_dir.generic_string());

			std::string wait_for = exec->time_interval ? exec->time_interval.value() : "600s";

			if (vm->run("/bin/bash", {guest_script_file.generic_string()}, time_to_seconds(wait_for)) != 0) {
				throw std::runtime_error("Bash command failed");
			}
			vm->remove_from_guest(guest_script_dir);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(exec, vm));
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
		std::throw_with_nested(ActionException(set, vm));
	}

}

void VisitorInterpreter::visit_copy(std::shared_ptr<VmController> vm, std::shared_ptr<Copy> copy) {
	try {
		fs::path from = visit_word(vm, copy->from);
		fs::path to = visit_word(vm, copy->to);

		std::string from_to = copy->is_to_guest() ? "to" : "from";

		print("Copying ", from, " ", from_to, " vm ", vm->name(), " in directory ", to);

		if (!vm->is_running()) {
			throw std::runtime_error(fmt::format("vm is not running"));
		}

		if (!vm->is_additions_installed()) {
			throw std::runtime_error(fmt::format("guest additions are not installed"));
		}

		std::string wait_for = copy->time_interval ? copy->time_interval.value() : "600s";

		if(copy->is_to_guest()) {
			if (from.is_relative()) {
				from = copy->t.pos().file.parent_path() / from;
			}
			vm->copy_to_guest(from, to, time_to_seconds(wait_for));
		} else {
			if (to.is_relative()) {
				to = copy->t.pos().file.parent_path() / to;
			}
			vm->copy_from_guest(from, to, time_to_seconds(wait_for));;
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(copy, vm));
	}

}

void VisitorInterpreter::visit_macro_call(std::shared_ptr<VmController> vm, std::shared_ptr<MacroCall> macro_call) {
	print("Calling macro ", macro_call->name().value(), " on vm ", vm->name());
	//push new ctx
	StackEntry new_ctx(true);

	for (size_t i = 0; i < macro_call->params.size(); ++i) {
		auto value = visit_word(vm, macro_call->params[i]);
		new_ctx.define(macro_call->macro->params[i].value(), value);
	}

	local_vars.push_back(new_ctx);
	coro::Finally finally([&] {
		local_vars.pop_back();
	});

	visit_action_block(vm, macro_call->macro->action_block->action);
}

void VisitorInterpreter::visit_if_clause(std::shared_ptr<VmController> vm, std::shared_ptr<IfClause> if_clause) {
	bool expr_result;
	try {
		expr_result = visit_expr(vm, if_clause->expr);
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(if_clause, vm));
	}
	//everything else should be caught at test level
	if (expr_result) {
		return visit_action(vm, if_clause->if_action);
	} else if (if_clause->has_else()) {
		return visit_action(vm, if_clause->else_action);
	}

}

void VisitorInterpreter::visit_for_clause(std::shared_ptr<VmController> vm, std::shared_ptr<ForClause> for_clause) {
	StackEntry new_ctx(false);
	local_vars.push_back(new_ctx);
	size_t ctx_position = local_vars.size() - 1;
	coro::Finally finally([&]{
		local_vars.pop_back();
	});
	for (auto i = for_clause->start(); i <= for_clause->finish(); i++) {
		local_vars[ctx_position].define(for_clause->counter.value(), std::to_string(i));
		try {
			visit_action(vm, for_clause->cycle_body);
		} catch (const CycleControlException& cycle_control) {
			if (cycle_control.token.type() == Token::category::break_) {
				break;
			} else if (cycle_control.token.type() == Token::category::continue_) {
				continue;
			} else {
				throw std::runtime_error(std::string("Unknown cycle control command: ") + cycle_control.token.value());
			}
		}
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
	} else if (auto p = std::dynamic_pointer_cast<Factor<Check>>(factor)) {
		return p->is_negated() ^ visit_check(vm, p->factor);
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

	for (auto it = local_vars.rbegin(); it != local_vars.rend(); ++it) {
		if (it->is_defined(var)) {
			return it->ref(var);
		}
		if (it->is_terminate) {
			break;
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
	if (comparison->op().type() == Token::category::GREATER) {
		if (!is_number(left)) {
			throw std::runtime_error(std::string(*comparison->left) + " is not an integer number");
		}
		if (!is_number(right)) {
			throw std::runtime_error(std::string(*comparison->right) + " is not an integer number");
		}

		return std::stoul(left) > std::stoul(right);

	} else if (comparison->op().type() == Token::category::LESS) {
		if (!is_number(left)) {
			throw std::runtime_error(std::string(*comparison->left) + " is not an integer number");
		}
		if (!is_number(right)) {
			throw std::runtime_error(std::string(*comparison->right) + " is not an integer number");
		}

		return std::stoul(left) < std::stoul(right);

	} else if (comparison->op().type() == Token::category::EQUAL) {
		if (!is_number(left)) {
			throw std::runtime_error(std::string(*comparison->left) + " is not an integer number");
		}
		if (!is_number(right)) {
			throw std::runtime_error(std::string(*comparison->right) + " is not an integer number");
		}

		return std::stoul(left) == std::stoul(right);

	} else if (comparison->op().type() == Token::category::STRGREATER) {
		return left > right;
	} else if (comparison->op().type() == Token::category::STRLESS) {
		return left < right;
	} else if (comparison->op().type() == Token::category::STREQUAL) {
		return left == right;
	} else {
		throw std::runtime_error("Unknown comparison op");
	}
}

bool VisitorInterpreter::visit_check(std::shared_ptr<VmController> vm, std::shared_ptr<Check> check) {
	try {
		auto text = visit_word(vm, check->text_word);

		std::string print_str = std::string("Checking ") + text;
		nlohmann::json params = {};

		for (auto it = check->params.begin(); it != check->params.end();) {
			if (it == check->params.begin()) {
				print_str += "(";
			}

			auto value = visit_word(vm, (*it)->right);
			params[(*it)->left.value()] = value;
			print_str += (*it)->left.value() + "=" + value;

			if (++it != check->params.end()) {
				print_str += ", ";
				continue;
			}
			print_str += ")";
		}

		print_str += std::string(" on vm ") + vm->name();
		print(print_str);
		return vm->check(text, params);
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(check, vm));
	}
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

	//Check also dvd contingency
	return (old_config == new_config);
}


std::string VisitorInterpreter::test_cksum(std::shared_ptr<Test> test) {
	VisitorCksum visitor(reg);
	return std::to_string(visitor.visit(test));
}

std::string VisitorInterpreter::cksum(std::shared_ptr<FlashDriveController> fd) {
	auto config = fd->get_config();
	std::string cksum_input = fd->name() + std::to_string(config.at("size").get<uint32_t>()) + config.at("fs").get<std::string>();
	if (fd->has_folder()) {
		cksum_input += directory_signature(config.at("folder").get<std::string>());
	}

	std::hash<std::string> h;
	return std::to_string(h(cksum_input));
}

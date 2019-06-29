
#include "VisitorInterpreter.hpp"
#include "VisitorCksum.hpp"

#include "coro/Finally.h"
#include <fmt/format.h>
#include <fstream>
#include <thread>
#include <wildcards.hpp>

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
	exclude = config.at("exclude").get<std::string>();

	charmap.insert({
		{'0', {"ZERO"}},
		{'1', {"ONE"}},
		{'2', {"TWO"}},
		{'3', {"THREE"}},
		{'4', {"FOUR"}},
		{'5', {"FIVE"}},
		{'6', {"SIX"}},
		{'7', {"SEVEN"}},
		{'8', {"EIGHT"}},
		{'9', {"NINE"}},
		{')', {"LEFTSHIFT", "ZERO"}},
		{'!', {"LEFTSHIFT", "ONE"}},
		{'@', {"LEFTSHIFT", "TWO"}},
		{'#', {"LEFTSHIFT", "THREE"}},
		{'$', {"LEFTSHIFT", "FOUR"}},
		{'%', {"LEFTSHIFT", "FIVE"}},
		{'^', {"LEFTSHIFT", "SIX"}},
		{'&', {"LEFTSHIFT", "SEVEN"}},
		{'*', {"LEFTSHIFT", "EIGHT"}},
		{'(', {"LEFTSHIFT", "NINE"}},
		{'a', {"A"}},
		{'b', {"B"}},
		{'c', {"C"}},
		{'d', {"D"}},
		{'e', {"E"}},
		{'f', {"F"}},
		{'g', {"G"}},
		{'h', {"H"}},
		{'i', {"I"}},
		{'j', {"J"}},
		{'k', {"K"}},
		{'l', {"L"}},
		{'m', {"M"}},
		{'n', {"N"}},
		{'o', {"O"}},
		{'p', {"P"}},
		{'q', {"Q"}},
		{'r', {"R"}},
		{'s', {"S"}},
		{'t', {"T"}},
		{'u', {"U"}},
		{'v', {"V"}},
		{'w', {"W"}},
		{'x', {"X"}},
		{'y', {"Y"}},
		{'z', {"Z"}},
		{'A', {"LEFTSHIFT", "A"}},
		{'B', {"LEFTSHIFT", "B"}},
		{'C', {"LEFTSHIFT", "C"}},
		{'D', {"LEFTSHIFT", "D"}},
		{'E', {"LEFTSHIFT", "E"}},
		{'F', {"LEFTSHIFT", "F"}},
		{'G', {"LEFTSHIFT", "G"}},
		{'H', {"LEFTSHIFT", "H"}},
		{'I', {"LEFTSHIFT", "I"}},
		{'J', {"LEFTSHIFT", "J"}},
		{'K', {"LEFTSHIFT", "K"}},
		{'L', {"LEFTSHIFT", "L"}},
		{'M', {"LEFTSHIFT", "M"}},
		{'N', {"LEFTSHIFT", "N"}},
		{'O', {"LEFTSHIFT", "O"}},
		{'P', {"LEFTSHIFT", "P"}},
		{'Q', {"LEFTSHIFT", "Q"}},
		{'R', {"LEFTSHIFT", "R"}},
		{'S', {"LEFTSHIFT", "S"}},
		{'T', {"LEFTSHIFT", "T"}},
		{'U', {"LEFTSHIFT", "U"}},
		{'V', {"LEFTSHIFT", "V"}},
		{'W', {"LEFTSHIFT", "W"}},
		{'X', {"LEFTSHIFT", "X"}},
		{'Y', {"LEFTSHIFT", "Y"}},
		{'Z', {"LEFTSHIFT", "Z"}},
		{'-', {"MINUS"}},
		{'_', {"LEFTSHIFT", "MINUS"}},
		{'=', {"EQUALSIGN"}},
		{'+', {"LEFTSHIFT", "EQUALSIGN"}},
		{'\'', {"APOSTROPHE"}},
		{'\"', {"LEFTSHIFT", "APOSTROPHE"}},
		{'\\', {"BACKSLASH"}},
		{'\n', {"ENTER"}},
		{'\t', {"TAB"}},
		{'|', {"LEFTSHIFT", "BACKSLASH"}},
		{',', {"COMMA"}},
		{'<', {"LEFTSHIFT", "COMMA"}},
		{'.', {"DOT"}},
		{'>', {"LEFTSHIFT", "DOT"}},
		{'/', {"SLASH"}},
		{'?', {"LEFTSHIFT", "SLASH"}},
		{';', {"SEMICOLON"}},
		{':', {"LEFTSHIFT", "SEMICOLON"}},
		{'[', {"LEFTBRACE"}},
		{'{', {"LEFTSHIFT", "LEFTBRACE"}},
		{']', {"RIGHTBRACE"}},
		{'}', {"LEFTSHIFT", "RIGHTBRACE"}},
		{'`', {"GRAVE"}},
		{'~', {"LEFTSHIFT", "GRAVE"}},
		{' ', {"SPACE"}}
	});
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

void VisitorInterpreter::resolve_tests(const std::vector<std::shared_ptr<AST::Test>>& tests_queue) {
	//Check every test
	for (auto test: tests_queue) {
		//1) If it's cached, just place it in the corresponding queue
		bool is_cached = true;

		for (auto parent: test->parents) {
			for (auto it: tests_to_run) {
				if (it->name.value() == parent->name.value()) {
					is_cached = false;
					break;
				}
			}
			if (!is_cached) {
				break;
			}
		}

		for (auto vmc: reg.get_all_vmcs(test)) {
			if (vmc->vm->is_defined() &&
				check_config_relevance(vmc->vm->get_config(), nlohmann::json::parse(vmc->get_metadata("vm_config"))) &&
				(file_signature(vmc->vm->get_config().at("iso").get<std::string>()) == vmc->get_metadata("dvd_signature")) &&
				vmc->has_snapshot(test->name.value()) &&
				(vmc->get_snapshot_cksum(test->name.value()) == test_cksum(test)))
			{
				continue;
			}
			is_cached = false;
		}

		if (is_cached) {
			up_to_date_tests.push_back(test);
		} else {
			//For now that's all
			tests_to_run.push_back(test);
		}
	}
}

void VisitorInterpreter::setup_vars(std::shared_ptr<Program> program) {
	std::vector<std::shared_ptr<AST::Test>> tests_queue; //temporary, only needed for general execution plan

	//Need to check that we don't have duplicates
	//And we can't use std::set because we need to
	//keep the order of the tests

	for (auto stmt: program->stmts) {
		if (auto p = std::dynamic_pointer_cast<Stmt<Test>>(stmt)) {
			auto test = p->stmt;

			//So for every test
			//we need to check if it's suitable for test spec
			//if it is - push back to list and remove all the parents duplicates

			if (test_spec.length() && !wildcards::match(test->name.value(), test_spec)) {
				continue;
			}

			if (exclude.length() && wildcards::match(test->name.value(), exclude)) {
				continue;
			}
			concat_unique(tests_queue, reg.get_test_path(test));
		} else if (auto p = std::dynamic_pointer_cast<Stmt<Controller>>(stmt)) {
			if (p->stmt->t.type() == Token::category::flash) {
				flash_drives.push_back(p->stmt);
			}
		}
	}

	resolve_tests(tests_queue);
	reset_cache();

	auto tests_num = tests_to_run.size() + up_to_date_tests.size();
	if (tests_num != 0) {
		progress_step = (float)100 / tests_num;
	} else {
		progress_step = 100;
	}
}


void VisitorInterpreter::reset_cache() {
	for (auto test: tests_to_run) {
		for (auto vmc: reg.get_all_vmcs(test)) {
			if (vmc->vm->is_defined()) {
				vmc->set_metadata("vm_current_state", "");
			}
		}
	}
}

void VisitorInterpreter::visit(std::shared_ptr<Program> program) {
	start_timestamp = std::chrono::system_clock::now();

	setup_vars(program);

	//Create flash drives
	for (auto fd: flash_drives) {
		visit_flash(fd);
	}

	if ((tests_to_run.size() + up_to_date_tests.size()) == 0) {
		std::cout << "There's no tests to run\n";
		return;
	}

	for (auto test: up_to_date_tests) {
		current_progress += progress_step;
		print("Test ", test->name.value(), " is up-to-date, skipping...");
	}

	while (!tests_to_run.empty()) {
		auto front = tests_to_run.front();
		tests_to_run.pop_front();
		visit_test(front);
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
		if (!fd->cache_enabled() || !fd->is_cksum_ok()) {
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
					current_progress += progress_step;
					print("Skipping test ", test->name.value(), " because his parent ", parent->name.value(), " failed");
					failed_tests.push_back(test);
					return;
				}
			}
		}

		//Ok, we're not cached and we need to run the test

		std::cout<< "Preparing the environment for the test " << test->name.value() << std::endl;

		//First we need to invalidate corresponding snapshots if they exist

		for (auto vmc: reg.get_all_vmcs(test)) {
			if (vmc->has_snapshot(test->name.value())) {
				vmc->delete_snapshot_with_children(test->name.value());
			}
		}

		//we need to get all the vms in the correct state
		//vms from parents - rollback them to parents if we need to
		//We need to do it only if our current state is not the parent
		for (auto parent: test->parents) {
			for (auto vmc: reg.get_all_vmcs(parent)) {
				if (vmc->get_metadata("vm_current_state") != parent->name.value()) {
					print("Restoring snapshot ", parent->name.value(), " for virtual machine ", vmc->vm->name());
					vmc->restore_snapshot(parent->name.value());
				}
			}
		}

		//new vms - install

		for (auto vmc: reg.get_all_vmcs(test)) {
			//check if it's a new one
			auto is_new = true;
			for (auto parent: test->parents) {
				auto parent_vmcs = reg.get_all_vmcs(parent);
				if (parent_vmcs.find(vmc) != parent_vmcs.end()) {
					//not new, go to the next vmc
					is_new = false;
					break;
				}
			}

			if (is_new) {
				print("Creating machine ", vmc->vm->name());
				vmc->create_vm();
			}
		}

		for (auto parent: test->parents) {
			for (auto vmc: reg.get_all_vmcs(parent)) {
				if (vmc->vm->state() == VmState::Suspended) {
					vmc->vm->resume();
				}
			}
		}

		std::cout<< "Running test " << test->name.value() << std::endl;

		//Everything is in the right state so we could actually do the test
		visit_command_block(test->cmd_block);

		//But that's not everything - we need to create according snapshots to all included vms

		for (auto vmc: reg.get_all_vmcs(test)) {
			if (vmc->vm->state() == VmState::Running) {
				vmc->vm->suspend();
			}
		}

		for (auto vmc: reg.get_all_vmcs(test)) {
			print("Taking snapshot ", test->name.value(), " for virtual machine ", vmc->vm->name());
			vmc->create_snapshot(test->name.value(), test_cksum(test), true); //true for now
		}

		//We need to check if we need to stop all the vms
		//VMS should be stopped if we don't need them anymore
		//and this could happen only if there's no children tests
		//ahead

		bool need_to_stop = true;

		for (auto it: tests_to_run) {
			for (auto parent: it->parents) {
				if (parent->name.value() == test->name.value()) {
					need_to_stop = false;
					break;
				}
			}
			if (need_to_stop) {
				break;
			}
		}

		if (need_to_stop) {
			stop_all_vms(test);
		}

		current_progress += progress_step;
		print("Test ", test->name.value(), " PASSED");
		succeeded_tests.push_back(test);

	} catch (const InterpreterException& error) {
		std::cout << error << std::endl;
		current_progress += progress_step;
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
		auto vmc = reg.vmcs.find(vm_token.value());
		visit_action(vmc->second, cmd->action);
	}
}


void VisitorInterpreter::visit_action_block(std::shared_ptr<VmController> vmc, std::shared_ptr<ActionBlock> action_block) {
	for (auto action: action_block->actions) {
		visit_action(vmc, action);
	}
}

void VisitorInterpreter::visit_action(std::shared_ptr<VmController> vmc, std::shared_ptr<IAction> action) {
	if (auto p = std::dynamic_pointer_cast<Action<Type>>(action)) {
		return visit_type(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Wait>>(action)) {
		return visit_wait(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Press>>(action)) {
		return visit_press(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Plug>>(action)) {
		return visit_plug(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Start>>(action)) {
		return visit_start(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Stop>>(action)) {
		return visit_stop(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Shutdown>>(action)) {
		return visit_shutdown(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Exec>>(action)) {
		return visit_exec(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Copy>>(action)) {
		return visit_copy(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<MacroCall>>(action)) {
		return visit_macro_call(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<IfClause>>(action)) {
		return visit_if_clause(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<ForClause>>(action)) {
		return visit_for_clause(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<CycleControl>>(action)) {
		throw CycleControlException(p->action->t);
	} else if (auto p = std::dynamic_pointer_cast<Action<ActionBlock>>(action)) {
		return visit_action_block(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<Action<Empty>>(action)) {
		return;
	} else {
		throw std::runtime_error("Unknown action");
	}
}

void VisitorInterpreter::visit_type(std::shared_ptr<VmController> vmc, std::shared_ptr<Type> type) {
	try {
		std::string text = visit_word(vmc, type->text_word);
		print("Typing ", text, " on virtual machine ", vmc->vm->name());
		for (auto c: text) {
			auto buttons = charmap.find(c);
			if (buttons == charmap.end()) {
				throw std::runtime_error("Unknown character to type");
			}
			vmc->vm->press(buttons->second);
			std::this_thread::sleep_for(std::chrono::milliseconds(30));
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(type, vmc));
	}
}

void VisitorInterpreter::visit_wait(std::shared_ptr<VmController> vmc, std::shared_ptr<Wait> wait) {
	try {
		std::string text = "";
		if (wait->text_word) {
			text = visit_word(vmc, wait->text_word);
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

			auto value = visit_word(vmc, (*it)->right);
			params[(*it)->left.value()] = value;
			print_str += (*it)->left.value() + "=" + value;

			if (++it != wait->params.end()) {
				print_str += ", ";
				continue;
			}
			print_str += ")";
		}

		print_str += std::string(" on virtual machine ") + vmc->vm->name();
		if (wait->time_interval) {
			print_str += " for " + wait->time_interval.value();
		}

		print(print_str);

		if (!wait->text_word) {
			return sleep(wait->time_interval.value());
		}

		std::string wait_for = wait->time_interval ? wait->time_interval.value() : "1m";

		auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(time_to_seconds(wait_for));

		while (std::chrono::system_clock::now() < deadline) {
			auto screenshot = vmc->vm->screenshot();
			if (shit.stink_even_stronger(screenshot, text)) {
				return;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}

		throw std::runtime_error("Wait timeout");
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(wait, vmc));
	}

}

void VisitorInterpreter::visit_press(std::shared_ptr<VmController> vmc, std::shared_ptr<Press> press) {
	try {
		for (auto key_spec: press->keys) {
			visit_key_spec(vmc, key_spec);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(press, vmc));
	}
}

void VisitorInterpreter::visit_key_spec(std::shared_ptr<VmController> vmc, std::shared_ptr<KeySpec> key_spec) {
	uint32_t times = key_spec->get_times();

	std::string print_str = std::string("Pressing button ") + key_spec->get_buttons_str();

	if (times > 1) {
		print_str += std::string(" ") + std::to_string(times) + " times ";
	}

	print_str += std::string(" on virtual machine ") + vmc->vm->name();

	print(print_str);

	for (uint32_t i = 0; i < times; i++) {
		vmc->vm->press(key_spec->get_buttons());
	}
}

void VisitorInterpreter::visit_plug(std::shared_ptr<VmController> vmc, std::shared_ptr<Plug> plug) {
	try {
		if (plug->type.value() == "nic") {
			return visit_plug_nic(vmc, plug);
		} else if (plug->type.value() == "link") {
			return visit_plug_link(vmc, plug);
		} else if (plug->type.value() == "dvd") {
			return visit_plug_dvd(vmc, plug);
		} else if (plug->type.value() == "flash") {
			if(plug->is_on()) {
				return plug_flash(vmc, plug);
			} else {
				return unplug_flash(vmc, plug);
			}
		} else {
			throw std::runtime_error(std::string("unknown hardware type to plug/unplug: ") +
				plug->type.value());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(plug, vmc));
	}
}

void VisitorInterpreter::visit_plug_nic(std::shared_ptr<VmController> vmc, std::shared_ptr<Plug> plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vmc while semantic analisys
	auto nic = plug->name_token.value();
	auto nics = vmc->vm->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("specified nic {} is not present in this virtual machine", nic));
	}

	if (vmc->vm->state() != VmState::Stopped) {
		throw std::runtime_error(fmt::format("virtual machine is running, but must be stopeed"));
	}

	if (vmc->vm->is_nic_plugged(nic) == plug->is_on()) {
		if (plug->is_on()) {
			throw std::runtime_error(fmt::format("specified nic {} is already plugged in this virtual machine", nic));
		} else {
			throw std::runtime_error(fmt::format("specified nic {} is not unplugged from this virtual machine", nic));
		}
	}

	std::string plug_unplug = plug->is_on() ? "plugging" : "unplugging";
	print(plug_unplug, " nic ", nic, " on virtual machine ", vmc->vm->name());

	vmc->vm->set_nic(nic, plug->is_on());
}

void VisitorInterpreter::visit_plug_link(std::shared_ptr<VmController> vmc, std::shared_ptr<Plug> plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vmc while semantic analisys

	auto nic = plug->name_token.value();
	auto nics = vmc->vm->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("the nic for specified link {} is not present in this virtual machine", nic));
	}

	if (!vmc->vm->is_nic_plugged(nic)) {
		throw std::runtime_error(fmt::format("the nic for specified link {} is unplugged, you must to plug it first", nic));
	}

	if (plug->is_on() == vmc->vm->is_link_plugged(nic)) {
		if (plug->is_on()) {
			throw std::runtime_error(fmt::format("specified link {} is already plugged in this virtual machine", nic));
		} else {
			throw std::runtime_error(fmt::format("specified link {} is already unplugged from this virtual machine", nic));
		}
	}

	std::string plug_unplug = plug->is_on() ? "plugging" : "unplugging";
	print(plug_unplug, " link ", nic, " on virtual machine ", vmc->vm->name());

	vmc->vm->set_link(nic, plug->is_on());
}

void VisitorInterpreter::plug_flash(std::shared_ptr<VmController> vmc, std::shared_ptr<Plug> plug) {
	auto fd = reg.fds.find(plug->name_token.value())->second; //should always be found
	print("Plugging flash drive ", fd->name(), " in virtual machine ", vmc->vm->name());
	if (vmc->vm->is_flash_plugged(fd)) {
		throw std::runtime_error(fmt::format("specified flash {} is already plugged into this virtual machine", fd->name()));
	}

	vmc->vm->plug_flash_drive(fd);
}

void VisitorInterpreter::unplug_flash(std::shared_ptr<VmController> vmc, std::shared_ptr<Plug> plug) {
	auto fd = reg.fds.find(plug->name_token.value())->second; //should always be found
	print("Unlugging flash drive ", fd->name(), " from virtual machine ", vmc->vm->name());
	if (!vmc->vm->is_flash_plugged(fd)) {
		throw std::runtime_error(fmt::format("specified flash {} is already unplugged from this virtual machine", fd->name()));
	}

	vmc->vm->unplug_flash_drive(fd);
}

void VisitorInterpreter::visit_plug_dvd(std::shared_ptr<VmController> vmc, std::shared_ptr<Plug> plug) {
	if (plug->is_on()) {
		if (vmc->vm->is_dvd_plugged()) {
			throw std::runtime_error(fmt::format("some dvd is already plugged"));
		}

		fs::path path = visit_word(vmc, plug->path);
		print("Plugging dvd ", path, " in virtual machine ", vmc->vm->name());
		if (path.is_relative()) {
			path = plug->t.pos().file.parent_path() / path;
		}
		vmc->vm->plug_dvd(path);
	} else {
		if (!vmc->vm->is_dvd_plugged()) {
			// throw std::runtime_error(fmt::format("dvd is already unplugged"));
			// это нормально, потому что поведение отличается от гипервизора к гипервизору
			// иногда у ОС получается открыть дисковод, иногда - нет
			return;
		}

		print("Unplugging dvd from virtual machine ", vmc->vm->name());
		vmc->vm->unplug_dvd();
	}
}

void VisitorInterpreter::visit_start(std::shared_ptr<VmController> vmc, std::shared_ptr<Start> start) {
	try {
		print("Starting virtual machine ", vmc->vm->name());
		vmc->vm->start();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(start, vmc));
	}
}

void VisitorInterpreter::visit_stop(std::shared_ptr<VmController> vmc, std::shared_ptr<Stop> stop) {
	try {
		print("Stopping virtual machine ", vmc->vm->name());
		vmc->vm->stop();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(stop, vmc));

	}
}

void VisitorInterpreter::visit_shutdown(std::shared_ptr<VmController> vmc, std::shared_ptr<Shutdown> shutdown) {
	try {
		print("Shutting down virtual machine ", vmc->vm->name());
		vmc->vm->power_button();
		std::string wait_for = shutdown->time_interval ? shutdown->time_interval.value() : "1m";
		auto deadline = std::chrono::system_clock::now() +  std::chrono::seconds(time_to_seconds(wait_for));
		while (std::chrono::system_clock::now() < deadline) {
			if (vmc->vm->state() == VmState::Stopped) {
				return;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
		}
		throw std::runtime_error("Shutdown timeout");
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(shutdown, vmc));

	}
}

void VisitorInterpreter::visit_exec(std::shared_ptr<VmController> vmc, std::shared_ptr<Exec> exec) {
	try {
		print("Executing ", exec->process_token.value(), " command on virtual machine ", vmc->vm->name());

		if (vmc->vm->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		if (!vmc->vm->is_additions_installed()) {
			throw std::runtime_error(fmt::format("guest additions is not installed"));
		}

		if (exec->process_token.value() == "bash") {
			//In future this should be a function

			std::string script = "set -e; set -o pipefail; set -x;";
			script += visit_word(vmc, exec->commands);
			script.erase(std::remove(script.begin(), script.end(), '\r'), script.end());

			//copy the script to tmp folder
			std::hash<std::string> h;

			std::string hash = std::to_string(h(script));

			fs::path host_script_dir = fs::temp_directory_path();
			fs::path guest_script_dir = fs::path("/tmp");

			fs::path host_script_file = host_script_dir / std::string(hash + ".sh");
			fs::path guest_script_file = guest_script_dir / std::string(hash + ".sh");
			std::ofstream script_stream(host_script_file, std::ios::binary);
			if (!script_stream.is_open()) {
				throw std::runtime_error(fmt::format("Can't open tmp file for writing the script"));
			}

			script_stream << script;
			script_stream.close();

			vmc->vm->copy_to_guest(host_script_file, guest_script_file, 5); //5 seconds should be enough to pass any script

			fs::remove(host_script_file.generic_string());

			std::string wait_for = exec->time_interval ? exec->time_interval.value() : "600s";

			if (vmc->vm->run("/bin/bash", {guest_script_file.generic_string()}, time_to_seconds(wait_for)) != 0) {
				throw std::runtime_error("Bash command failed");
			}
			vmc->vm->remove_from_guest(guest_script_file);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(exec, vmc));
	}
}

void VisitorInterpreter::visit_copy(std::shared_ptr<VmController> vmc, std::shared_ptr<Copy> copy) {
	try {
		fs::path from = visit_word(vmc, copy->from);
		fs::path to = visit_word(vmc, copy->to);

		std::string from_to = copy->is_to_guest() ? "to" : "from";

		print("Copying ", from, " ", from_to, " virtual machine ", vmc->vm->name(), " in directory ", to);

		if (vmc->vm->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		if (!vmc->vm->is_additions_installed()) {
			throw std::runtime_error(fmt::format("guest additions are not installed"));
		}

		std::string wait_for = copy->time_interval ? copy->time_interval.value() : "600s";

		if(copy->is_to_guest()) {
			if (from.is_relative()) {
				from = copy->t.pos().file.parent_path() / from;
			}
			vmc->vm->copy_to_guest(from, to, time_to_seconds(wait_for));
		} else {
			if (to.is_relative()) {
				to = copy->t.pos().file.parent_path() / to;
			}
			vmc->vm->copy_from_guest(from, to, time_to_seconds(wait_for));;
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(copy, vmc));
	}

}

void VisitorInterpreter::visit_macro_call(std::shared_ptr<VmController> vmc, std::shared_ptr<MacroCall> macro_call) {
	print("Calling macro ", macro_call->name().value(), " on virtual machine ", vmc->vm->name());
	//push new ctx
	StackEntry new_ctx(true);

	for (size_t i = 0; i < macro_call->params.size(); ++i) {
		auto value = visit_word(vmc, macro_call->params[i]);
		new_ctx.define(macro_call->macro->params[i].value(), value);
	}

	local_vars.push_back(new_ctx);
	coro::Finally finally([&] {
		local_vars.pop_back();
	});

	visit_action_block(vmc, macro_call->macro->action_block->action);
}

void VisitorInterpreter::visit_if_clause(std::shared_ptr<VmController> vmc, std::shared_ptr<IfClause> if_clause) {
	bool expr_result;
	try {
		expr_result = visit_expr(vmc, if_clause->expr);
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(if_clause, vmc));
	}
	//everything else should be caught at test level
	if (expr_result) {
		return visit_action(vmc, if_clause->if_action);
	} else if (if_clause->has_else()) {
		return visit_action(vmc, if_clause->else_action);
	}

}

void VisitorInterpreter::visit_for_clause(std::shared_ptr<VmController> vmc, std::shared_ptr<ForClause> for_clause) {
	StackEntry new_ctx(false);
	local_vars.push_back(new_ctx);
	size_t ctx_position = local_vars.size() - 1;
	coro::Finally finally([&]{
		local_vars.pop_back();
	});
	for (auto i = for_clause->start(); i <= for_clause->finish(); i++) {
		local_vars[ctx_position].define(for_clause->counter.value(), std::to_string(i));
		try {
			visit_action(vmc, for_clause->cycle_body);
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

bool VisitorInterpreter::visit_expr(std::shared_ptr<VmController> vmc, std::shared_ptr<IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<Expr<BinOp>>(expr)) {
		return visit_binop(vmc, p->expr);
	} else if (auto p = std::dynamic_pointer_cast<Expr<IFactor>>(expr)) {
		return visit_factor(vmc, p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

bool VisitorInterpreter::visit_binop(std::shared_ptr<VmController> vmc, std::shared_ptr<BinOp> binop) {
	auto left = visit_expr(vmc, binop->left);
	auto right = visit_expr(vmc, binop->right);

	if (binop->op().type() == Token::category::AND) {
		return left && right;
	} else if (binop->op().type() == Token::category::OR) {
		return left || right;
	} else {
		throw std::runtime_error("Unknown binop operation");
	}
}

bool VisitorInterpreter::visit_factor(std::shared_ptr<VmController> vmc, std::shared_ptr<IFactor> factor) {
	if (auto p = std::dynamic_pointer_cast<Factor<Word>>(factor)) {
		return p->is_negated() ^ (bool)visit_word(vmc, p->factor).length();
	} else if (auto p = std::dynamic_pointer_cast<Factor<Comparison>>(factor)) {
		return p->is_negated() ^ visit_comparison(vmc, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<Check>>(factor)) {
		return p->is_negated() ^ visit_check(vmc, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<Factor<IExpr>>(factor)) {
		return p->is_negated() ^ visit_expr(vmc, p->factor);
	} else {
		throw std::runtime_error("Unknown factor type");
	}
}

std::string VisitorInterpreter::resolve_var(std::shared_ptr<VmController> vmc, const std::string& var) {
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

	if (vmc->vm->is_defined() && vmc->has_key(var)) {
		return vmc->get_metadata(var);
	}

	auto env_value = std::getenv(var.c_str());

	if (env_value == nullptr) {
		return "";
	}
	return env_value;
}

std::string VisitorInterpreter::visit_word(std::shared_ptr<VmController> vmc, std::shared_ptr<Word> word) {
	std::string result;

	for (auto part: word->parts) {
		if (part.type() == Token::category::dbl_quoted_string) {
			result += part.value().substr(1, part.value().length() - 2);
		} else if (part.type() == Token::category::var_ref) {
			result += resolve_var(vmc, part.value().substr(1, part.value().length() - 1));
		} else if (part.type() == Token::category::multiline_string) {
			result += part.value().substr(3, part.value().length() - 6);
		} else {
			throw std::runtime_error("Unknown word type");
		}
	}

	return result;
}

bool VisitorInterpreter::visit_comparison(std::shared_ptr<VmController> vmc, std::shared_ptr<Comparison> comparison) {
	auto left = visit_word(vmc, comparison->left);
	auto right = visit_word(vmc, comparison->right);
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

bool VisitorInterpreter::visit_check(std::shared_ptr<VmController> vmc, std::shared_ptr<Check> check) {
	try {
		auto text = visit_word(vmc, check->text_word);

		std::string print_str = std::string("Checking ") + text;
		nlohmann::json params = {};

		for (auto it = check->params.begin(); it != check->params.end();) {
			if (it == check->params.begin()) {
				print_str += "(";
			}

			auto value = visit_word(vmc, (*it)->right);
			params[(*it)->left.value()] = value;
			print_str += (*it)->left.value() + "=" + value;

			if (++it != check->params.end()) {
				print_str += ", ";
				continue;
			}
			print_str += ")";
		}

		print_str += std::string(" on virtual machine ") + vmc->vm->name();
		print(print_str);
		auto screenshot = vmc->vm->screenshot();
		return shit.stink_even_stronger(screenshot, text);
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(check, vmc));
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

	new_config.erase("iso");
	old_config.erase("iso");

	//Check also dvd contingency
	return (old_config == new_config);
}


std::string VisitorInterpreter::test_cksum(std::shared_ptr<Test> test) {
	VisitorCksum visitor(reg);
	return std::to_string(visitor.visit(test));
}

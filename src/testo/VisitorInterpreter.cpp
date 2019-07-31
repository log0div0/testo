
#include "VisitorInterpreter.hpp"
#include "VisitorCksum.hpp"

#include "coro/Finally.h"
#include <fmt/format.h>
#include <fstream>
#include <thread>
#include <wildcards.hpp>

using namespace std::chrono_literals;

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
	cache_miss_policy = config.at("cache_miss_policy").get<std::string>();
	test_spec = config.at("test_spec").get<std::string>();
	exclude = config.at("exclude").get<std::string>();
	invalidate = config.at("invalidate").get<std::string>();

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
	auto total_tests = succeeded_tests.size() + failed_tests.size() + up_to_date_tests.size() + ignored_tests.size();
	auto tests_durantion = std::chrono::system_clock::now() - start_timestamp;

	std::cout << "PROCESSED TOTAL " << total_tests << " TESTS IN " << duration_to_str(tests_durantion) << std::endl;
	std::cout << "UP TO DATE: " << up_to_date_tests.size() << std::endl;
	if (ignored_tests.size()) {
		std::cout << "LOST CACHE, BUT SKIPPED: " << ignored_tests.size() << std::endl;
		for (auto ignore: ignored_tests) {
			std::cout << "\t -" << ignore->name.value() << std::endl;
		}
	}
	std::cout << "RUN SUCCESSFULLY: " << succeeded_tests.size() << std::endl;
	std::cout << "FAILED: " << failed_tests.size() << std::endl;
	for (auto fail: failed_tests) {
		std::cout << "\t -" << fail->name.value() << std::endl;
	}
}

bool VisitorInterpreter::parent_is_ok(std::shared_ptr<AST::Test> test, std::shared_ptr<AST::Test> parent,
	std::list<std::shared_ptr<AST::Test>>::reverse_iterator begin,
	std::list<std::shared_ptr<AST::Test>>::reverse_iterator end)
{
	auto controllers = reg.get_all_controllers(test);
	auto all_parents = reg.get_test_path(test);

	bool result = false;

	for (auto rit = tests_to_run.rbegin(); rit != tests_to_run.rend(); ++rit) {
		if ((*rit)->name.value() == parent->name.value()) {
			//This parent is good
			result = true;
			break;
		}

		//If it's just another parent - we don't care
		bool another_parent = false;
		for (auto test_it: all_parents) {
			if (test_it->name.value() == (*rit)->name.value()) {
				another_parent = true;
				break;
			}
		}

		if (another_parent) {
			continue;
		}

		auto other_controllers = reg.get_all_controllers(*rit);
		if (std::find_first_of (controllers.begin(), controllers.end(), other_controllers.begin(), other_controllers.end()) != controllers.end()) {
			break;
		}
	}

	return result;
}

void VisitorInterpreter::build_test_plan(std::shared_ptr<AST::Test> test,
	std::list<std::shared_ptr<AST::Test>>& test_plan,
	std::list<std::shared_ptr<AST::Test>>::reverse_iterator begin,
	std::list<std::shared_ptr<AST::Test>>::reverse_iterator end)
{
	//we need to check could we start right away?

	for (auto parent: test->parents) {
		//for every parent we need to check, maybe we are already in the perfect position?
		//so starting from the end of tests_to_run, we move backwards
		//and we try to find the parent test

		if (!parent_is_ok(test, parent, begin, end) && !parent->snapshots_needed) {
			//New tests to run should be JUST before the parent
			std::list<std::shared_ptr<AST::Test>> new_tests_to_run;

			for (auto rit = begin; rit != end; ++rit) {
				if ((*rit)->name.value() == parent->name.value()) {
					begin = ++rit;
					break;
				}
			}

			build_test_plan(parent, test_plan, begin, end);
		}
	}
	test_plan.push_back(test);
}

bool VisitorInterpreter::is_cached(std::shared_ptr<AST::Test> test) const {
	for (auto parent: test->parents) {
		bool parent_cached = false;
		for (auto cached: up_to_date_tests) {
			if (parent->name.value() == cached->name.value()) {
				parent_cached = true;
				break;
			}
		}
		if (!parent_cached) {
			return false;
		}
	}

	for (auto controller: reg.get_all_controllers(test)) {
		if (controller->is_defined() &&
			controller->check_config_relevance() &&
			controller->has_snapshot(test->name.value()) &&
			(controller->get_snapshot_cksum(test->name.value()) == test_cksum(test)))
		{
			continue;
		}
		return false;
	}
	return true;
}

bool VisitorInterpreter::resolve_miss_cache_action(std::shared_ptr<AST::Test> test) const {
	if (cache_miss_policy.length()) {
		//cache miss policy is set
		if (cache_miss_policy == "skip_branch") {
			return false;
		} else if (cache_miss_policy == "accept") {
			return true;
		} else if (cache_miss_policy == "abort") {
			throw std::runtime_error(std::string("Test ") + test->name.value() + " lost cache, aborting");
		} else {
			throw std::runtime_error("Unknown cache_miss_policy"); //should never happen, just a failsafe
		}
	}

	//is some parent is ignored - we should ignore this one as well
	for (auto parent: test->parents) {
		for (auto ignored_test: ignored_tests) {
			if (parent == ignored_test) {
				return false;
			}
		}
	}

	//if at least one of the parents is not up-to-date, then it was scheduled to run and the user accepted its running
	//So no need to ask twice

	for (auto parent: test->parents) {
		bool found = false;

		for (auto up_to_date: up_to_date_tests) {
			if (parent == up_to_date) {
				found = true;
				break;
			}
		}
		if (!found) {
			return true;
		}
	}

	bool prompt_needed = false;

	for (auto controller: reg.get_all_controllers(test)) {
		if (controller->is_defined()) {
			if (!controller->check_config_relevance()) {
				prompt_needed = true;
				break;
			}

			if (controller->has_snapshot(test->name.value())) {
				if (controller->get_snapshot_cksum(test->name.value()) != test_cksum(test)) {
					prompt_needed = true;
					break;
				}
			}
		}
	}

	if (!prompt_needed) {
		return true;
	}

	std::string choice;
	std::cout << "Test " << test->name.value() << " lost its cache. It and all its children will run again" << std::endl;
	std::cout << "Do you confirm the running of the test? [Y/n]: ";
	std::getline(std::cin, choice);

	std::transform(choice.begin(), choice.end(), choice.begin(), ::toupper);

	if (!choice.length() || choice == "Y" || choice == "YES") {
		return true;
	}

	return false;
}

void VisitorInterpreter::check_up_to_date_tests(std::list<std::shared_ptr<AST::Test>>& tests_queue) {
	//Check every test
	for (auto test_it = tests_queue.begin(); test_it != tests_queue.end();) {
		if (is_cached(*test_it)) {
			up_to_date_tests.push_back(*test_it);
			tests_queue.erase(test_it++);
		} else {
			//prompt with what to do with the test
			if (!resolve_miss_cache_action(*test_it)) {
				ignored_tests.push_back(*test_it);
				tests_queue.erase(test_it++);
			} else {
				test_it++;
			}
		}
	}
}

void VisitorInterpreter::resolve_tests(const std::list<std::shared_ptr<AST::Test>>& tests_queue) {
	for (auto test: tests_queue) {
		for (auto controller: reg.get_all_controllers(test)) {
			if (controller->has_snapshot(test->name.value())) {
				controller->delete_snapshot_with_children(test->name.value());
			}
		}

		//Now the interesting part
		//We already have the logic involving current_state, so all we need to do...
		//is to fill up the test queue with intermediate tests
		std::list<std::shared_ptr<AST::Test>> test_plan;

		build_test_plan(test, test_plan, tests_to_run.rbegin(), tests_to_run.rend());

		//TODO: insert before last
		tests_to_run.insert(tests_to_run.end(), test_plan.begin(), test_plan.end());
	}
}

void VisitorInterpreter::setup_vars(std::shared_ptr<AST::Program> program) {
	std::list<std::shared_ptr<AST::Test>> tests_queue; //temporary, only needed for general execution plan

	//Need to check that we don't have duplicates
	//And we can't use std::set because we need to
	//keep the order of the tests

	for (auto stmt: program->stmts) {
		if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Test>>(stmt)) {
			auto test = p->stmt;

			//invalidate tests at request

			if (invalidate.length() && wildcards::match(test->name.value(), invalidate)) {
				for (auto controller: reg.get_all_controllers(test)) {
					if (controller->has_snapshot(test->name.value())) {
						controller->delete_snapshot_with_children(test->name.value());
					}
				}
			}

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
		} else if (auto p = std::dynamic_pointer_cast<AST::Stmt<AST::Controller>>(stmt)) {
			if (p->stmt->t.type() == Token::category::flash) {
				flash_drives.push_back(p->stmt);
			}
		}
	}

	check_up_to_date_tests(tests_queue);
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
		for (auto controller: reg.get_all_controllers(test)) {
			if (controller->is_defined()) {
				controller->set_metadata("current_state", "");
			}
		}
	}
}

void VisitorInterpreter::visit(std::shared_ptr<AST::Program> program) {
	start_timestamp = std::chrono::system_clock::now();

	setup_vars(program);

	if ((tests_to_run.size() + up_to_date_tests.size()) == 0) {
		std::cout << "There's no tests to run\n";
		return;
	}

	for (auto test: up_to_date_tests) {
		current_progress += progress_step;
		print("Test ", test->name.value(), " is up-to-date, skipping...");
	}

	std::cout << "TEST TO RUN\n";
	for (auto it: tests_to_run) {
		std::cout << it->name.value() << std::endl;
	}

	while (!tests_to_run.empty()) {
		auto front = tests_to_run.front();
		tests_to_run.pop_front();
		visit_test(front);
	}

	print_statistics();

	if (failed_tests.size()) {
		throw std::runtime_error("At least one of the tests failed");
	}
}

void VisitorInterpreter::visit_test(std::shared_ptr<AST::Test> test) {
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

		print("Preparing the environment for the test ", test->name.value());

		//we need to get all the vms in the correct state
		//vms from parents - rollback them to parents if we need to
		//We need to do it only if our current state is not the parent
		for (auto parent: test->parents) {
			for (auto controller: reg.get_all_controllers(parent)) {
				if (controller->get_metadata("current_state") != parent->name.value()) {
					print("Restoring snapshot ", parent->name.value(), " for entity ", controller->name());
					controller->restore_snapshot(parent->name.value());
				}
			}
		}

		//new vms - install

		for (auto controller: reg.get_all_controllers(test)) {
			//check if it's a new one
			auto is_new = true;
			for (auto parent: test->parents) {
				auto parent_controller = reg.get_all_controllers(parent);
				if (parent_controller.find(controller) != parent_controller.end()) {
					//not new, go to the next vmc
					is_new = false;
					break;
				}
			}

			if (is_new) {
				//Ok, here we need to do some refactoring
				//If the config is relevant, and the init snapshot is avaliable
				//we should restore init snapshot
				//Otherwise we're creating the controller and taking init snapshot
				if (controller->is_defined() &&
					controller->has_snapshot("_init") &&
					controller->check_config_relevance())
				{
					print("Restoring initial snapshot for entity ", controller->name());
					controller->restore_snapshot("_init");
				} else {
					print("Creating entity ", controller->name());
					controller->create();
					print("Taking initial snapshot for entity ", controller->name());
					controller->create_snapshot("_init", "", true);
					controller->set_metadata("current_state", "_init");
				}
			}
		}

		for (auto parent: test->parents) {
			for (auto vmc: reg.get_all_vmcs(parent)) {
				if (vmc->vm->state() == VmState::Suspended) {
					vmc->vm->resume();
				}
			}
		}

		print("Running test ", test->name.value());

		//Everything is in the right state so we could actually do the test
		visit_command_block(test->cmd_block);

		//But that's not everything - we need to create according snapshots to all included vms

		for (auto vmc: reg.get_all_vmcs(test)) {
			if (vmc->vm->state() == VmState::Running) {
				vmc->vm->suspend();
			}
		}

		for (auto controller: reg.get_all_controllers(test)) {
			if (!controller->has_snapshot(test->name.value())) {
				print("Taking snapshot ", test->name.value(), " for entity ", controller->name());
				controller->create_snapshot(test->name.value(), test_cksum(test), test->snapshots_needed);
			}
			controller->set_metadata("current_state", test->name.value());
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
			if (!need_to_stop) {
				break;
			}
		}

		if (need_to_stop) {
			stop_all_vms(test);
		}

		current_progress += progress_step;
		print("Test ", test->name.value(), " PASSED");

		for (auto it: up_to_date_tests) {
			if (it->name.value() == test->name.value()) {
				//already have that one
				return;
			}
		}

		for (auto it: succeeded_tests) {
			if (it->name.value() == test->name.value()) {
				//already have that one
				return;
			}
		}

		succeeded_tests.push_back(test);

	} catch (const InterpreterException& error) {
		std::cout << error << std::endl;
		current_progress += progress_step;
		print ("Test ", test->name.value(), " FAILED");

		bool already_failed = false;
		for (auto it: failed_tests) {
			if (it->name.value() == test->name.value()) {
				already_failed = true;
			}
		}

		if (!already_failed) {
			failed_tests.push_back(test);
		}

		if (stop_on_fail) {
			throw std::runtime_error("");
		}

		stop_all_vms(test);

	} //everything else is fatal and should be catched furter up
}

void VisitorInterpreter::visit_command_block(std::shared_ptr<AST::CmdBlock> block) {
	for (auto command: block->commands) {
		visit_command(command);
	}
}

void VisitorInterpreter::visit_command(std::shared_ptr<AST::Cmd> cmd) {
	for (auto vm_token: cmd->vms) {
		auto vmc = reg.vmcs.find(vm_token.value());
		visit_action(vmc->second, cmd->action);
	}
}


void VisitorInterpreter::visit_action_block(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::ActionBlock> action_block) {
	for (auto action: action_block->actions) {
		visit_action(vmc, action);
	}
}

void VisitorInterpreter::visit_action(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Abort>>(action)) {
		return visit_abort(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Print>>(action)) {
		return visit_print(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Type>>(action)) {
		return visit_type(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Wait>>(action)) {
		return visit_wait(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Press>>(action)) {
		return visit_press(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Plug>>(action)) {
		return visit_plug(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Start>>(action)) {
		return visit_start(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Stop>>(action)) {
		return visit_stop(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Shutdown>>(action)) {
		return visit_shutdown(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Exec>>(action)) {
		return visit_exec(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Copy>>(action)) {
		return visit_copy(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroCall>>(action)) {
		return visit_macro_call(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		return visit_if_clause(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		return visit_for_clause(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::CycleControl>>(action)) {
		throw CycleControlException(p->action->t);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		return visit_action_block(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Empty>>(action)) {
		return;
	} else {
		throw std::runtime_error("Unknown action");
	}
}

void VisitorInterpreter::visit_abort(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Abort> abort) {
	std::string message = visit_word(vmc, abort->message);
	throw AbortException(abort, vmc, message);
}

void VisitorInterpreter::visit_print(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Print> print_action) {
	try {
		std::string message = visit_word(vmc, print_action->message);
		print(vmc->name(), ": ", message.c_str());
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(print_action, vmc));
	}
}

void VisitorInterpreter::visit_type(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Type> type) {
	try {
		std::string text = visit_word(vmc, type->text_word);
		print("Typing ", text, " on virtual machine ", vmc->name());
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

void VisitorInterpreter::visit_wait(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Wait> wait) {
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

		print_str += std::string(" on virtual machine ") + vmc->name();
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
			auto start = std::chrono::high_resolution_clock::now();
			auto screenshot = vmc->vm->screenshot();
			if (shit.stink_even_stronger(screenshot, text)) {
				return;
			}
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time = end - start;
			// std::cout << "time = " << time.count() << " seconds" << std::endl;
			if (time < 1s) {
				std::this_thread::sleep_for(1s - time);
			}
		}

		throw std::runtime_error("Wait timeout");
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(wait, vmc));
	}

}

void VisitorInterpreter::visit_press(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Press> press) {
	try {
		for (auto key_spec: press->keys) {
			visit_key_spec(vmc, key_spec);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(press, vmc));
	}
}

void VisitorInterpreter::visit_key_spec(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::KeySpec> key_spec) {
	uint32_t times = key_spec->get_times();

	std::string print_str = std::string("Pressing button ") + key_spec->get_buttons_str();

	if (times > 1) {
		print_str += std::string(" ") + std::to_string(times) + " times ";
	}

	print_str += std::string(" on virtual machine ") + vmc->name();

	print(print_str);

	for (uint32_t i = 0; i < times; i++) {
		vmc->vm->press(key_spec->get_buttons());
	}
}

void VisitorInterpreter::visit_plug(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
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

void VisitorInterpreter::visit_plug_nic(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
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
	print(plug_unplug, " nic ", nic, " on virtual machine ", vmc->name());

	vmc->vm->set_nic(nic, plug->is_on());
}

void VisitorInterpreter::visit_plug_link(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
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
	print(plug_unplug, " link ", nic, " on virtual machine ", vmc->name());

	vmc->vm->set_link(nic, plug->is_on());
}

void VisitorInterpreter::plug_flash(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
	auto fdc = reg.fdcs.find(plug->name_token.value())->second; //should always be found
	print("Plugging flash drive ", fdc->name(), " in virtual machine ", vmc->name());
	if (vmc->vm->is_flash_plugged(fdc->fd)) {
		throw std::runtime_error(fmt::format("specified flash {} is already plugged into this virtual machine", fdc->name()));
	}

	vmc->vm->plug_flash_drive(fdc->fd);
}

void VisitorInterpreter::unplug_flash(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
	auto fdc = reg.fdcs.find(plug->name_token.value())->second; //should always be found
	print("Unlugging flash drive ", fdc->name(), " from virtual machine ", vmc->name());
	if (!vmc->vm->is_flash_plugged(fdc->fd)) {
		throw std::runtime_error(fmt::format("specified flash {} is already unplugged from this virtual machine", fdc->name()));
	}

	vmc->vm->unplug_flash_drive(fdc->fd);
}

void VisitorInterpreter::visit_plug_dvd(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
	if (plug->is_on()) {
		if (vmc->vm->is_dvd_plugged()) {
			throw std::runtime_error(fmt::format("some dvd is already plugged"));
		}

		fs::path path = visit_word(vmc, plug->path);
		print("Plugging dvd ", path, " in virtual machine ", vmc->name());
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

		print("Unplugging dvd from virtual machine ", vmc->name());
		vmc->vm->unplug_dvd();
	}
}

void VisitorInterpreter::visit_start(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Start> start) {
	try {
		print("Starting virtual machine ", vmc->name());
		vmc->vm->start();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(start, vmc));
	}
}

void VisitorInterpreter::visit_stop(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Stop> stop) {
	try {
		print("Stopping virtual machine ", vmc->name());
		vmc->vm->stop();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(stop, vmc));

	}
}

void VisitorInterpreter::visit_shutdown(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Shutdown> shutdown) {
	try {
		print("Shutting down virtual machine ", vmc->name());
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

void VisitorInterpreter::visit_exec(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Exec> exec) {
	try {
		print("Executing ", exec->process_token.value(), " command on virtual machine ", vmc->name());

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

void VisitorInterpreter::visit_copy(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Copy> copy) {
	try {
		fs::path from = visit_word(vmc, copy->from);
		fs::path to = visit_word(vmc, copy->to);

		std::string from_to = copy->is_to_guest() ? "to" : "from";

		print("Copying ", from, " ", from_to, " virtual machine ", vmc->name(), " in directory ", to);

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

void VisitorInterpreter::visit_macro_call(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MacroCall> macro_call) {
	print("Calling macro ", macro_call->name().value(), " on virtual machine ", vmc->name());
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

void VisitorInterpreter::visit_if_clause(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IfClause> if_clause) {
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

void VisitorInterpreter::visit_for_clause(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::ForClause> for_clause) {
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

bool VisitorInterpreter::visit_expr(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::BinOp>>(expr)) {
		return visit_binop(vmc, p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::IFactor>>(expr)) {
		return visit_factor(vmc, p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

bool VisitorInterpreter::visit_binop(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::BinOp> binop) {
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

bool VisitorInterpreter::visit_factor(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::IFactor> factor) {
	if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Word>>(factor)) {
		return p->is_negated() ^ (bool)visit_word(vmc, p->factor).length();
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Comparison>>(factor)) {
		return p->is_negated() ^ visit_comparison(vmc, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Check>>(factor)) {
		return p->is_negated() ^ visit_check(vmc, p->factor);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::IExpr>>(factor)) {
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

	if (vmc->is_defined() && vmc->has_user_key(var)) {
		return vmc->get_user_metadata(var);
	}

	auto env_value = std::getenv(var.c_str());

	if (env_value == nullptr) {
		return "";
	}
	return env_value;
}

std::string VisitorInterpreter::visit_word(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Word> word) {
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

bool VisitorInterpreter::visit_comparison(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Comparison> comparison) {
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

bool VisitorInterpreter::visit_check(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Check> check) {
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

		print_str += std::string(" on virtual machine ") + vmc->name();
		print(print_str);
		auto screenshot = vmc->vm->screenshot();
		return shit.stink_even_stronger(screenshot, text);
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(check, vmc));
	}
}

std::string VisitorInterpreter::test_cksum(std::shared_ptr<AST::Test> test) const {
	VisitorCksum visitor(reg);
	return std::to_string(visitor.visit(test));
}

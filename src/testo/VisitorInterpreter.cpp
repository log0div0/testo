
#include "VisitorInterpreter.hpp"
#include "IR/Program.hpp"
#include "Exceptions.hpp"

#include "coro/CheckPoint.h"
#include "coro/Timeout.h"
#include "utf8.hpp"
#include <fmt/format.h>
#include <fstream>
#include <thread>
#include <wildcards.hpp>
#include <rang.hpp>

using namespace std::chrono_literals;

static void sleep(const std::string& interval) {
	coro::Timer timer;
	timer.waitFor(std::chrono::milliseconds(time_to_milliseconds(interval)));
}

VisitorInterpreter::VisitorInterpreter(const nlohmann::json& config) {
	reporter = Reporter(config);

	stop_on_fail = config.at("stop_on_fail").get<bool>();
	assume_yes = config.at("assume_yes").get<bool>();
	invalidate = config.at("invalidate").get<std::string>();

	charmap.insert({
		{"0", {"ZERO"}},
		{"1", {"ONE"}},
		{"2", {"TWO"}},
		{"3", {"THREE"}},
		{"4", {"FOUR"}},
		{"5", {"FIVE"}},
		{"6", {"SIX"}},
		{"7", {"SEVEN"}},
		{"8", {"EIGHT"}},
		{"9", {"NINE"}},
		{")", {"LEFTSHIFT", "ZERO"}},
		{"!", {"LEFTSHIFT", "ONE"}},
		{"@", {"LEFTSHIFT", "TWO"}},
		{"#", {"LEFTSHIFT", "THREE"}},
		{"$", {"LEFTSHIFT", "FOUR"}},
		{"%", {"LEFTSHIFT", "FIVE"}},
		{"^", {"LEFTSHIFT", "SIX"}},
		{"&", {"LEFTSHIFT", "SEVEN"}},
		{"*", {"LEFTSHIFT", "EIGHT"}},
		{"(", {"LEFTSHIFT", "NINE"}},
		{"a", {"A"}},
		{"b", {"B"}},
		{"c", {"C"}},
		{"d", {"D"}},
		{"e", {"E"}},
		{"f", {"F"}},
		{"g", {"G"}},
		{"h", {"H"}},
		{"i", {"I"}},
		{"j", {"J"}},
		{"k", {"K"}},
		{"l", {"L"}},
		{"m", {"M"}},
		{"n", {"N"}},
		{"o", {"O"}},
		{"p", {"P"}},
		{"q", {"Q"}},
		{"r", {"R"}},
		{"s", {"S"}},
		{"t", {"T"}},
		{"u", {"U"}},
		{"v", {"V"}},
		{"w", {"W"}},
		{"x", {"X"}},
		{"y", {"Y"}},
		{"z", {"Z"}},
		{"A", {"LEFTSHIFT", "A"}},
		{"B", {"LEFTSHIFT", "B"}},
		{"C", {"LEFTSHIFT", "C"}},
		{"D", {"LEFTSHIFT", "D"}},
		{"E", {"LEFTSHIFT", "E"}},
		{"F", {"LEFTSHIFT", "F"}},
		{"G", {"LEFTSHIFT", "G"}},
		{"H", {"LEFTSHIFT", "H"}},
		{"I", {"LEFTSHIFT", "I"}},
		{"J", {"LEFTSHIFT", "J"}},
		{"K", {"LEFTSHIFT", "K"}},
		{"L", {"LEFTSHIFT", "L"}},
		{"M", {"LEFTSHIFT", "M"}},
		{"N", {"LEFTSHIFT", "N"}},
		{"O", {"LEFTSHIFT", "O"}},
		{"P", {"LEFTSHIFT", "P"}},
		{"Q", {"LEFTSHIFT", "Q"}},
		{"R", {"LEFTSHIFT", "R"}},
		{"S", {"LEFTSHIFT", "S"}},
		{"T", {"LEFTSHIFT", "T"}},
		{"U", {"LEFTSHIFT", "U"}},
		{"V", {"LEFTSHIFT", "V"}},
		{"W", {"LEFTSHIFT", "W"}},
		{"X", {"LEFTSHIFT", "X"}},
		{"Y", {"LEFTSHIFT", "Y"}},
		{"Z", {"LEFTSHIFT", "Z"}},

		{"а", {"F"}},
		{"б", {"COMMA"}},
		{"в", {"D"}},
		{"г", {"U"}},
		{"д", {"L"}},
		{"е", {"T"}},
		{"ё", {"GRAVE"}},
		{"ж", {"SEMICOLON"}},
		{"з", {"P"}},
		{"и", {"B"}},
		{"й", {"Q"}},
		{"к", {"R"}},
		{"л", {"K"}},
		{"м", {"V"}},
		{"н", {"Y"}},
		{"о", {"J"}},
		{"п", {"G"}},
		{"р", {"H"}},
		{"с", {"C"}},
		{"т", {"N"}},
		{"у", {"E"}},
		{"ф", {"A"}},
		{"х", {"LEFTBRACE"}},
		{"ц", {"W"}},
		{"ч", {"X"}},
		{"ш", {"I"}},
		{"щ", {"O"}},
		{"ъ", {"RIGHTBRACE"}},
		{"ы", {"S"}},
		{"ь", {"M"}},
		{"э", {"APOSTROPHE"}},
		{"ю", {"DOT"}},
		{"я", {"Z"}},

		{"А", {"LEFTSHIFT", "F"}},
		{"Б", {"LEFTSHIFT", "COMMA"}},
		{"В", {"LEFTSHIFT", "D"}},
		{"Г", {"LEFTSHIFT", "U"}},
		{"Д", {"LEFTSHIFT", "L"}},
		{"Е", {"LEFTSHIFT", "T"}},
		{"Ё", {"LEFTSHIFT", "GRAVE"}},
		{"Ж", {"LEFTSHIFT", "SEMICOLON"}},
		{"З", {"LEFTSHIFT", "P"}},
		{"И", {"LEFTSHIFT", "B"}},
		{"Й", {"LEFTSHIFT", "Q"}},
		{"К", {"LEFTSHIFT", "R"}},
		{"Л", {"LEFTSHIFT", "K"}},
		{"М", {"LEFTSHIFT", "V"}},
		{"Н", {"LEFTSHIFT", "Y"}},
		{"О", {"LEFTSHIFT", "J"}},
		{"П", {"LEFTSHIFT", "G"}},
		{"Р", {"LEFTSHIFT", "H"}},
		{"С", {"LEFTSHIFT", "C"}},
		{"Т", {"LEFTSHIFT", "N"}},
		{"У", {"LEFTSHIFT", "E"}},
		{"Ф", {"LEFTSHIFT", "A"}},
		{"Х", {"LEFTSHIFT", "LEFTBRACE"}},
		{"Ц", {"LEFTSHIFT", "W"}},
		{"Ч", {"LEFTSHIFT", "X"}},
		{"Ш", {"LEFTSHIFT", "I"}},
		{"Щ", {"LEFTSHIFT", "O"}},
		{"Ъ", {"LEFTSHIFT", "RIGHTBRACE"}},
		{"Ы", {"LEFTSHIFT", "S"}},
		{"Ь", {"LEFTSHIFT", "M"}},
		{"Э", {"LEFTSHIFT", "APOSTROPHE"}},
		{"Ю", {"LEFTSHIFT", "DOT"}},
		{"Я", {"LEFTSHIFT", "Z"}},

		{"-", {"MINUS"}},
		{"_", {"LEFTSHIFT", "MINUS"}},
		{"=", {"EQUALSIGN"}},
		{"+", {"LEFTSHIFT", "EQUALSIGN"}},
		{"\'", {"APOSTROPHE"}},
		{"\"", {"LEFTSHIFT", "APOSTROPHE"}},
		{"\\", {"BACKSLASH"}},
		{"\n", {"ENTER"}},
		{"\t", {"TAB"}},
		{"|", {"LEFTSHIFT", "BACKSLASH"}},
		{",", {"COMMA"}},
		{"<", {"LEFTSHIFT", "COMMA"}},
		{".", {"DOT"}},
		{">", {"LEFTSHIFT", "DOT"}},
		{"/", {"SLASH"}},
		{"?", {"LEFTSHIFT", "SLASH"}},
		{";", {"SEMICOLON"}},
		{":", {"LEFTSHIFT", "SEMICOLON"}},
		{"[", {"LEFTBRACE"}},
		{"{", {"LEFTSHIFT", "LEFTBRACE"}},
		{"]", {"RIGHTBRACE"}},
		{"}", {"LEFTSHIFT", "RIGHTBRACE"}},
		{"`", {"GRAVE"}},
		{"~", {"LEFTSHIFT", "GRAVE"}},
		{" ", {"SPACE"}}
	});
}

bool VisitorInterpreter::parent_is_ok(std::shared_ptr<IR::Test> test, std::shared_ptr<IR::Test> parent,
	std::list<std::shared_ptr<IR::Test>>::reverse_iterator begin,
	std::list<std::shared_ptr<IR::Test>>::reverse_iterator end)
{
	auto controllers = test->get_all_controllers();
	auto all_parents = IR::Test::get_test_path(test);

	bool result = false;

	for (auto rit = tests_to_run.rbegin(); rit != tests_to_run.rend(); ++rit) {
		if ((*rit)->name() == parent->name()) {
			//This parent is good
			result = true;
			break;
		}

		//If it's just another parent - we don't care
		bool another_parent = false;
		for (auto test_it: all_parents) {
			if (test_it->name() == (*rit)->name()) {
				another_parent = true;
				break;
			}
		}

		if (another_parent) {
			continue;
		}

		auto other_controllers = (*rit)->get_all_controllers();
		if (std::find_first_of (controllers.begin(), controllers.end(), other_controllers.begin(), other_controllers.end()) != controllers.end()) {
			break;
		}
	}

	return result;
}

void VisitorInterpreter::build_test_plan(std::shared_ptr<IR::Test> test,
	std::list<std::shared_ptr<IR::Test>>& test_plan,
	std::list<std::shared_ptr<IR::Test>>::reverse_iterator begin,
	std::list<std::shared_ptr<IR::Test>>::reverse_iterator end)
{
	//we need to check could we start right away?

	for (auto parent: test->parents) {
		//for every parent we need to check, maybe we are already in the perfect position?
		//so starting from the end of tests_to_run, we move backwards
		//and we try to find the parent test

		if (!parent_is_ok(test, parent, begin, end) && !parent->snapshots_needed()) {
			//New tests to run should be JUST before the parent
			std::list<std::shared_ptr<IR::Test>> new_tests_to_run;

			for (auto rit = begin; rit != end; ++rit) {
				if ((*rit)->name() == parent->name()) {
					begin = ++rit;
					break;
				}
			}

			build_test_plan(parent, test_plan, begin, end);
		}
	}
	test_plan.push_back(test);
}

bool VisitorInterpreter::is_cached(std::shared_ptr<IR::Test> test) const {
	for (auto parent: test->parents) {
		bool parent_cached = false;
		for (auto cached: up_to_date_tests) {
			if (parent->name() == cached->name()) {
				parent_cached = true;
				break;
			}
		}
		if (!parent_cached) {
			return false;
		}
	}

	//check networks aditionally
	for (auto network: test->get_all_networks()) {
		if (network->is_defined() &&
			network->check_config_relevance())
		{
			continue;
		}
		return false;
	}

	for (auto controller: test->get_all_controllers()) {
		if (controller->is_defined() &&
			controller->has_snapshot("_init") &&
			controller->check_metadata_version() &&
			controller->check_config_relevance() &&
			controller->has_snapshot(test->name()) &&
			(controller->get_snapshot_cksum(test->name()) == test->cksum))
		{
			continue;
		}
		return false;
	}
	return true;
}

bool VisitorInterpreter::is_cache_miss(std::shared_ptr<IR::Test> test) const {
	auto all_parents = IR::Test::get_test_path(test);

	for (auto parent: all_parents) {
		for (auto cache_missed_test: cache_missed_tests) {
			if (parent == cache_missed_test) {
				return false;
			}
		}
	}

	//check networks aditionally
	for (auto netc: test->get_all_networks()) {
		if (netc->is_defined()) {
			if (!netc->check_config_relevance()) {
				return true;
			}
		}
	}

	for (auto controller: test->get_all_controllers()) {
		if (controller->is_defined()) {
			if (controller->has_snapshot(test->name())) {
				if (controller->get_snapshot_cksum(test->name()) != test->cksum) {
					return true;
				}
				if (!controller->check_config_relevance()) {
					return true;
				}
			}
		}
	}

	return false;
}

void VisitorInterpreter::check_up_to_date_tests(std::list<std::shared_ptr<IR::Test>>& tests_queue) {
	//Check every test
	for (auto test_it = tests_queue.begin(); test_it != tests_queue.end();) {
		if (is_cached(*test_it)) {
			up_to_date_tests.push_back(*test_it);
			tests_queue.erase(test_it++);
		} else {
			if (is_cache_miss(*test_it)) {
				cache_missed_tests.push_back(*test_it);
			}
			test_it++;
		}
	}
}

void VisitorInterpreter::resolve_tests(const std::list<std::shared_ptr<IR::Test>>& tests_queue) {
	for (auto test: tests_queue) {
		for (auto controller: test->get_all_controllers()) {
			if (controller->is_defined() && controller->has_snapshot(test->name())) {
				controller->delete_snapshot_with_children(test->name());
			}
		}

		//Now the interesting part
		//We already have the logic involving current_state, so all we need to do...
		//is to fill up the test queue with intermediate tests
		std::list<std::shared_ptr<IR::Test>> test_plan;

		build_test_plan(test, test_plan, tests_to_run.rbegin(), tests_to_run.rend());

		//TODO: insert before last
		tests_to_run.insert(tests_to_run.end(), test_plan.begin(), test_plan.end());
	}
}

void VisitorInterpreter::setup_vars() {
	std::list<std::shared_ptr<IR::Test>> tests_queue; //temporary, only needed for general execution plan

	//Need to check that we don't have duplicates
	//And we can't use std::set because we need to
	//keep the order of the tests

	for (auto& test: IR::program->all_selected_tests) {

		//invalidate tests at request

		if (invalidate.length() && wildcards::match(test->name(), invalidate)) {
			for (auto controller: test->get_all_controllers()) {
				if (controller->is_defined() && controller->has_snapshot(test->name())) {
					controller->delete_snapshot_with_children(test->name());
				}
			}
		}

		concat_unique(tests_queue, IR::Test::get_test_path(test));
	}

	check_up_to_date_tests(tests_queue);

	if (!assume_yes && cache_missed_tests.size()) {
		std::cout << "Some tests have lost their cache:" << std::endl;

		for (auto cache_missed: cache_missed_tests) {
			std::cout << "\t- " << cache_missed->name() << std::endl;
		}

		std::cout << "Do you confirm running them and all their children? [y/N]: ";
		std::string choice;
		std::getline(std::cin, choice);

		std::transform(choice.begin(), choice.end(), choice.begin(), ::toupper);

		if (choice != "Y" && choice != "YES") {
			throw std::runtime_error("Aborted");
		}
	}

	resolve_tests(tests_queue);
	reset_cache();
}

void VisitorInterpreter::reset_cache() {
	for (auto test: tests_to_run) {
		for (auto controller: test->get_all_controllers()) {
			if (controller->is_defined()) {
				controller->current_state = "";
			}
		}
	}
}

void VisitorInterpreter::visit() {
	setup_vars();

	reporter.init(tests_to_run, up_to_date_tests);

	while (!tests_to_run.empty()) {
		auto front = tests_to_run.front();
		tests_to_run.pop_front();
		visit_test(front);
	}

	reporter.finish();
	if (reporter.failed_tests.size()) {
		throw std::runtime_error("At least one of the tests failed");
	}
}

void VisitorInterpreter::visit_test(std::shared_ptr<IR::Test> test) {
	try {
		current_test = nullptr;
		//Ok, we're not cached and we need to run the test
		reporter.prepare_environment();
		//Check if one of the parents failed. If it did, just fail
		for (auto parent: test->parents) {
			for (auto failed: reporter.failed_tests) {
				if (parent->name() == failed->name()) {
					reporter.skip_failed_test(parent->name());
					return;
				}
			}
		}

		//we need to get all the vms in the correct state
		//vms from parents - rollback them to parents if we need to
		//We need to do it only if our current state is not the parent
		for (auto parent: test->parents) {
			for (auto controller: parent->get_all_controllers()) {
				if (controller->current_state != parent->name()) {
					reporter.restore_snapshot(controller, parent->name());
					controller->restore_snapshot(parent->name());
					coro::CheckPoint();
				}
			}
		}

		//check all the networks

		for (auto netc: test->get_all_networks()) {
			if (netc->is_defined() &&
				netc->check_config_relevance())
			{
				continue;
			}
			netc->create();
		}

		//new vms - install

		for (auto controller: test->get_all_controllers()) {
			//check if it's a new one
			auto is_new = true;
			for (auto parent: test->parents) {
				auto parent_controller = parent->get_all_controllers();
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
					controller->check_metadata_version() &&
					controller->check_config_relevance())
				{
					reporter.restore_snapshot(controller, "initial");
					controller->restore_snapshot("_init");
					coro::CheckPoint();
				} else {
					reporter.create_controller(controller);
					controller->create();

					reporter.take_snapshot(controller, "initial");
					controller->create_snapshot("_init", "", true);
					controller->current_state = "_init";
					coro::CheckPoint();
				}
			}
		}

		for (auto parent: test->parents) {
			for (auto vmc: parent->get_all_machines()) {
				if (vmc->vm()->state() == VmState::Suspended) {
					vmc->vm()->resume();
				}
			}
		}

		reporter.run_test();

		//Everything is in the right state so we could actually do the test
		{
			StackPusher<VisitorInterpreter> pusher(this, test->stack);
			current_test = test;
			visit_command_block(test->ast_node->cmd_block);
		}

		//But that's not everything - we need to create according snapshots to all included vms
		for (auto vmc: test->get_all_machines()) {
			if (vmc->vm()->state() == VmState::Running) {
				vmc->vm()->suspend();
			}
		}

		//we need to take snapshots in the right order
		//1) all the vms - so we could check that all the fds are unplugged
		for (auto controller: test->get_all_machines()) {
			if (!controller->has_snapshot(test->name())) {
				reporter.take_snapshot(controller, test->name());
				controller->create_snapshot(test->name(), test->cksum, test->snapshots_needed());
				coro::CheckPoint();
			}
			controller->current_state = test->name();
		}

		//2) all the fdcs - the rest
		for (auto controller: test->get_all_flash_drives()) {
			if (!controller->has_snapshot(test->name())) {
				reporter.take_snapshot(controller, test->name());
				controller->create_snapshot(test->name(), test->cksum, test->snapshots_needed());
				coro::CheckPoint();
			}
			controller->current_state = test->name();
		}

		//We need to check if we need to stop all the vms
		//VMS should be stopped if we don't need them anymore
		//and this could happen only if there's no children tests
		//ahead

		bool need_to_stop = true;

		for (auto it: tests_to_run) {
			for (auto parent: it->parents) {
				if (parent->name() == test->name()) {
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
		reporter.test_passed();

	} catch (const Exception& error) {
		std::stringstream ss;
		ss << error << std::endl;
		reporter.test_failed(ss.str());

		if (stop_on_fail) {
			throw std::runtime_error("");
		}

		stop_all_vms(test);
	}
}

void VisitorInterpreter::visit_command_block(std::shared_ptr<AST::CmdBlock> block) {
	for (auto command: block->commands) {
		visit_command(command);
	}
}

void VisitorInterpreter::visit_command(std::shared_ptr<AST::Cmd> cmd) {
	current_controller = nullptr;
	if (current_controller = IR::program->get_machine_or_null(cmd->entity.value())) {
		visit_action_vm(cmd->action);
	} else if (current_controller = IR::program->get_flash_drive_or_null(cmd->entity.value())) {
		visit_action_fd(cmd->action);
	} else {
		throw std::runtime_error("Should never happen");
	}
}


void VisitorInterpreter::visit_action_block(std::shared_ptr<AST::ActionBlock> action_block) {
	for (auto action: action_block->actions) {
		visit_action(action);
	}
}

void VisitorInterpreter::visit_action(std::shared_ptr<AST::IAction> action) {
	if (std::dynamic_pointer_cast<IR::Machine>(current_controller)) {
		visit_action_vm(action);
	} else {
		visit_action_fd(action);
	}
}

void VisitorInterpreter::visit_action_vm(std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Abort>>(action)) {
		visit_abort({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Print>>(action)) {
		visit_print({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Type>>(action)) {
		visit_type({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Wait>>(action)) {
		visit_wait({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		visit_sleep({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Press>>(action)) {
		visit_press({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Hold>>(action)) {
		visit_hold({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Release>>(action)) {
		visit_release({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Mouse>>(action)) {
		visit_mouse({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Plug>>(action)) {
		visit_plug({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Start>>(action)) {
		visit_start({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Stop>>(action)) {
		visit_stop({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Shutdown>>(action)) {
		visit_shutdown({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Exec>>(action)) {
		visit_exec({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Copy>>(action)) {
		visit_copy({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroCall>>(action)) {
		visit_macro_call(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		visit_if_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		visit_for_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::CycleControl>>(action)) {
		throw CycleControlException(p->action->t);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		visit_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Empty>>(action)) {
		;
	} else {
		throw std::runtime_error("Should never happen");
	}

	coro::CheckPoint();
}

void VisitorInterpreter::visit_action_fd(std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Abort>>(action)) {
		visit_abort({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Print>>(action)) {
		visit_print({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		visit_sleep({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Copy>>(action)) {
		visit_copy({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		visit_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Empty>>(action)) {
		;
	/*} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroCall>>(action)) {
		visit_macro_call(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		visit_if_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		visit_for_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::CycleControl>>(action)) {
		throw CycleControlException(p->action->t);*/
	}  else {
		throw std::runtime_error("Should never happen");
	}

	coro::CheckPoint();
}

void VisitorInterpreter::visit_abort(const IR::Abort& abort) {
	throw AbortException(abort.ast_node, current_controller, abort.message());
}

void VisitorInterpreter::visit_print(const IR::Print& print) {
	try {
		reporter.print(current_controller, print.message());
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(print.ast_node, current_controller));
	}
}

void VisitorInterpreter::visit_type(const IR::Type& type) {
	try {
		std::string text = type.text();
		if (text.size() == 0) {
			return;
		}

		std::string interval = type.interval();

		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.type(vmc, text, interval);

		for (auto c: utf8::split_to_chars(text)) {
			auto buttons = charmap.find(c);
			if (buttons == charmap.end()) {
				throw std::runtime_error("Unknown character to type");
			}
			vmc->press(buttons->second);
			timer.waitFor(std::chrono::milliseconds(time_to_milliseconds(interval)));
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(type.ast_node, current_controller));
	}
}

bool VisitorInterpreter::visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr, stb::Image& screenshot) {
	if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::ISelectable>>(select_expr)) {
		return visit_detect_selectable(p->select_expr, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::SelectBinOp>>(select_expr)) {
		return visit_detect_binop(p->select_expr, screenshot);
	} else {
		throw std::runtime_error("Unknown select expression type");
	}
}

js::Value VisitorInterpreter::eval_js(const std::string& script, stb::Image& screenshot) {
	try {
		js_current_ctx.reset(new js::Context(&screenshot));
		return js_current_ctx->eval(script);
	} catch (const nn::ContinueError& error) {
		throw error;
	}
	catch(const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Error while executing javascript selection"));
	}
}

bool VisitorInterpreter::visit_detect_js(const IR::SelectJS& js, stb::Image& screenshot) {
	auto value = eval_js(js.script(), screenshot);

	if (value.is_bool()) {
		return (bool)value;
	} else {
	 	throw std::runtime_error("Can't process return value type. We expect a single boolean");
	}
}

nn::Point VisitorInterpreter::visit_select_js(const IR::SelectJS& js, stb::Image& screenshot) {
	auto value = eval_js(js.script(), screenshot);

	if (value.is_object() && !value.is_array()) {
		auto x_prop = value.get_property_str("x");
		if (x_prop.is_undefined()) {
			throw std::runtime_error("Object doesn't have the x propery");
		}

		auto y_prop = value.get_property_str("y");
		if (y_prop.is_undefined()) {
			throw std::runtime_error("Object doesn't have the y propery");
		}

		nn::Point point;
		point.x = x_prop;
		point.y = y_prop;
		return point;
	} else {
		throw std::runtime_error("Can't process return value type. We expect a single object");
	}
}

nn::Tensor VisitorInterpreter::visit_select_text(const IR::SelectText& text, stb::Image& screenshot) {
	auto parsed = text.text();
	return  nn::find_text(&screenshot).match(parsed);
}

bool VisitorInterpreter::visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable, stb::Image& screenshot) {
	bool is_negated = selectable->is_negated();

	if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectText>>(selectable)) {
		return is_negated ^ (bool)visit_select_text({p->selectable, stack}, screenshot).size();
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(selectable)) {
		return is_negated ^ visit_detect_js({p->selectable, stack}, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectParentedExpr>>(selectable)) {
		return is_negated ^ visit_detect_expr(p->selectable->select_expr, screenshot);
	}  else {
		throw std::runtime_error("Unknown selectable type");
	}
}

bool VisitorInterpreter::visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop, stb::Image& screenshot) {
	auto left_value = visit_detect_expr(binop->left, screenshot);
	if (binop->t.type() == Token::category::double_ampersand) {
		if (!left_value) {
			return false;
		} else {
			return left_value && visit_detect_expr(binop->right, screenshot);
		}
	} else if (binop->t.type() == Token::category::double_vertical_bar) {
		if (left_value) {
			return true;
		} else {
			return left_value || visit_detect_expr(binop->right, screenshot);
		}
	} else {
		throw std::runtime_error("Unknown binop operation");
	}
}

void VisitorInterpreter::visit_sleep(const IR::Sleep& sleep) {
	reporter.sleep(current_controller, sleep.timeout());
	::sleep(sleep.timeout());
}

void VisitorInterpreter::visit_wait(const IR::Wait& wait) {
	try {
		std::string wait_for = wait.timeout();
		std::string interval_str = wait.interval();
		auto interval = std::chrono::milliseconds(time_to_milliseconds(interval_str));
		auto text = wait.select_expr();

		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.wait(vmc, text, wait_for, interval_str);

		auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(time_to_milliseconds(wait_for));

		while (std::chrono::system_clock::now() < deadline) {
			auto start = std::chrono::high_resolution_clock::now();
			auto screenshot = vmc->vm()->screenshot();

			if (visit_detect_expr(wait.ast_node->select_expr, screenshot)) {
				return;
			}

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time = end - start;
			//std::cout << "time = " << time.count() << " seconds" << std::endl;
			if (interval > end - start) {
				timer.waitFor(interval - (end - start));
			} else {
				coro::CheckPoint();
			}
		}

		if (reporter.report_screenshots) {
			reporter.save_screenshot(vmc);
		}
		throw std::runtime_error("Timeout");

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(wait.ast_node, current_controller));
	}
}

void VisitorInterpreter::visit_press(const IR::Press& press) {
	try {
		std::string interval = press.interval();
		auto press_interval = time_to_milliseconds(interval);

		for (auto key_spec: press.ast_node->keys) {
			visit_key_spec(key_spec, press_interval);
			timer.waitFor(std::chrono::milliseconds(press_interval));
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(press.ast_node, current_controller));
	}
}

void VisitorInterpreter::visit_hold(const IR::Hold& hold) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.hold_key(vmc, std::string(*hold.ast_node->combination));
		vmc->hold(hold.buttons());
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(hold.ast_node, current_controller));
	}
}

void VisitorInterpreter::visit_release(const IR::Release& release) {
	try {
		auto buttons = release.buttons();

		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		if (buttons.size()) {
			reporter.release_key(vmc, std::string(*release.ast_node->combination));
			vmc->release(release.buttons());
		} else {
			reporter.release_key(vmc);
			vmc->release();
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(release.ast_node, current_controller));
	}
}


void VisitorInterpreter::visit_mouse(const IR::Mouse& mouse) {
	if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseMoveClick>>(mouse.ast_node->event)) {
		return visit_mouse_move_click({p->event, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseHold>>(mouse.ast_node->event)) {
		return visit_mouse_hold({p->event, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseRelease>>(mouse.ast_node->event)) {
		return visit_mouse_release({p->event, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseWheel>>(mouse.ast_node->event)) {
		throw std::runtime_error("Not implemented yet");
	} else {
		throw std::runtime_error("Unknown mouse actions");
	}
}

void VisitorInterpreter::visit_mouse_hold(const IR::MouseHold& mouse_hold) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.mouse_hold(vmc, mouse_hold.button());
		if (mouse_hold.button() == "lbtn") {
			vmc->mouse_hold({MouseButton::Left});
		} else if (mouse_hold.button() == "rbtn") {
			vmc->mouse_hold({MouseButton::Right});
		} else if (mouse_hold.button() == "mbtn") {
			vmc->mouse_hold({MouseButton::Middle});
		} else {
			throw std::runtime_error("Unknown mouse button: " + mouse_hold.button());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_hold.ast_node, current_controller));
	}
}

void VisitorInterpreter::visit_mouse_release(const IR::MouseRelease& mouse_release) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.mouse_release(vmc);
		vmc->mouse_release();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_release.ast_node, current_controller));
	}
}

void VisitorInterpreter::visit_mouse_wheel(std::shared_ptr<AST::MouseWheel> mouse_wheel) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.mouse_wheel(vmc, mouse_wheel->direction.value());

		if (mouse_wheel->direction.value() == "up") {
			vmc->mouse_press({MouseButton::WheelUp});
		} else if (mouse_wheel->direction.value() == "down") {
			vmc->mouse_press({MouseButton::WheelDown});
		} else {
			throw std::runtime_error("Unknown wheel direction");
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_wheel, current_controller));
	}
}

void VisitorInterpreter::visit_mouse_move_click(const IR::MouseMoveClick& mouse_move_click) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.mouse_move_click(vmc, mouse_move_click.event_type());

		if (mouse_move_click.ast_node->object) {
			if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseCoordinates>>(mouse_move_click.ast_node->object)) {
				visit_mouse_move_coordinates({p->target, stack});
			} else if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseSelectable>>(mouse_move_click.ast_node->object)) {
				visit_mouse_move_selectable({p->target, stack});
			} else {
				throw std::runtime_error("Unknown mouse move target");
			}
		} else {
			reporter.mouse_no_object();
		}

		if (mouse_move_click.event_type() == "move") {
			return;
		}

		if (mouse_move_click.event_type() == "click" || mouse_move_click.event_type() == "lclick") {
			vmc->mouse_press({MouseButton::Left});
		} else if (mouse_move_click.event_type() == "rclick") {
			vmc->mouse_press({MouseButton::Right});
		} else if (mouse_move_click.event_type() == "mclick") {
			vmc->mouse_press({MouseButton::Middle});
		} else if (mouse_move_click.event_type() == "dclick") {
			vmc->mouse_press({MouseButton::Left});
			timer.waitFor(std::chrono::milliseconds(60));
			vmc->mouse_press({MouseButton::Left});
		} else {
			throw std::runtime_error("Unsupported click type");
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_move_click.ast_node, current_controller));
	}
}


nn::Tensor VisitorInterpreter::visit_mouse_specifier_from(
	std::shared_ptr<AST::MouseAdditionalSpecifier> specifier,
	const nn::Tensor& input)
{
	auto name = specifier->name.value();
	auto arg = std::stoul(specifier->arg.value()); //should never fail since we have semantic checks

	if (name == "from_top") {
		return input.from_top(arg);
	} else if (name == "from_bottom") {
		return input.from_bottom(arg);
	} else if (name == "from_left") {
		return input.from_left(arg);
	} else if (name == "from_right") {
		return input.from_right(arg);
	}

	throw std::runtime_error("Should not be there");
}

nn::Point VisitorInterpreter::visit_mouse_specifier_centering(
	std::shared_ptr<AST::MouseAdditionalSpecifier> specifier,
	const nn::Tensor& input)
{
	auto name = specifier->name.value();

	if (name == "left_bottom") {
		return input.left_bottom();
	} else if (name == "left_center") {
		return input.left_center();
	} else if (name == "left_top") {
		return input.left_top();
	} else if (name == "center_bottom") {
		return input.center_bottom();
	} else if (name == "center") {
		return input.center();
	} else if (name == "center_top") {
		return input.center_top();
	} else if (name == "right_bottom") {
		return input.right_bottom();
	} else if (name == "right_center") {
		return input.right_center();
	} else if (name == "right_top") {
		return input.right_top();
	}

	throw std::runtime_error("Uknown center specifier");
}

nn::Point VisitorInterpreter::visit_mouse_specifier_default_centering(const nn::Tensor& input) {
	return input.center();
}

nn::Point VisitorInterpreter::visit_mouse_specifier_moving(
	std::shared_ptr<AST::MouseAdditionalSpecifier> specifier,
	const nn::Point& input)
{
	auto name = specifier->name.value();
	auto arg = std::stoul(specifier->arg.value()); //should never fail since we have semantic checks

	if (name == "move_left") {
		return input.move_left(arg);
	} else if (name == "move_right") {
		return input.move_right(arg);
	} else if (name == "move_up") {
		return input.move_up(arg);
	} else if (name == "move_down") {
		return input.move_down(arg);
	}

	throw std::runtime_error("Should not be there");
}


nn::Point VisitorInterpreter::visit_mouse_additional_specifiers(
	const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers,
	const nn::Tensor& input_)
{
	size_t index = 0;

	nn::Tensor input = input_;

	if ((specifiers.size() > index) && specifiers[index]->is_from()) {
		input = visit_mouse_specifier_from(specifiers[index], input);
		index++;
	}

	nn::Point result;

	if (specifiers.size() > index && specifiers[index]->is_centering()) {
		result = visit_mouse_specifier_centering(specifiers[index], input);
		index++;
	} else {
		result = visit_mouse_specifier_default_centering(input);
	}

	for (size_t i = index; i < specifiers.size(); ++i) {
		result = visit_mouse_specifier_moving(specifiers[i], result);
	}

	return result;
}

void VisitorInterpreter::visit_mouse_move_selectable(const IR::MouseSelectable& mouse_selectable)
{
	std::string timeout = mouse_selectable.timeout();
	std::string where_to_go = mouse_selectable.where_to_go();

	for (auto specifier: mouse_selectable.ast_node->specifiers) {
		where_to_go += std::string(*specifier);
	}

	auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
	reporter.mouse_move_click_selectable(vmc, where_to_go, timeout);

	auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(time_to_milliseconds(timeout));

	while (std::chrono::system_clock::now() < deadline) {
		auto start = std::chrono::high_resolution_clock::now();
		auto screenshot = vmc->vm()->screenshot();
		try {
			nn::Point point;
			if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(mouse_selectable.ast_node->selectable)) {
				point = visit_select_js({p->selectable, stack}, screenshot);
			} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectText>>(mouse_selectable.ast_node->selectable)) {
				auto ocr_found = visit_select_text({p->selectable, stack}, screenshot);
				//each specifier can throw an exception if something goes wrong.
				point = visit_mouse_additional_specifiers(mouse_selectable.ast_node->specifiers, ocr_found);
			}
			vmc->vm()->mouse_move_abs(point.x, point.y);
			return;
		} catch (const nn::ContinueError& error) {
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time = end - start;
			if (time < 1s) {
				timer.waitFor(std::chrono::duration_cast<std::chrono::milliseconds>(1s - time));
			} else {
				coro::CheckPoint();
			}
			continue;
		}
	}

	if (reporter.report_screenshots) {
		reporter.save_screenshot(vmc);
	}

	throw std::runtime_error("Timeout");
}

void VisitorInterpreter::visit_mouse_move_coordinates(const IR::MouseCoordinates& coordinates)
{
	auto dx = coordinates.x();
	auto dy = coordinates.y();
	auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
	reporter.mouse_move_click_coordinates(vmc, dx, dy);
	if ((dx[0] == '+') || (dx[0] == '-')) {
		vmc->vm()->mouse_move_rel("x", std::stoi(dx));
	} else {
		vmc->vm()->mouse_move_abs("x", std::stoul(dx));
	}

	if ((dy[0] == '+') || (dy[0] == '-')) {
		vmc->vm()->mouse_move_rel("y", std::stoi(dy));
	} else {
		vmc->vm()->mouse_move_abs("y", std::stoul(dy));
	}
}

void VisitorInterpreter::visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec, uint32_t interval) {
	uint32_t times = key_spec->get_times();

	auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
	reporter.press_key(vmc, *key_spec->combination, times);

	for (uint32_t i = 0; i < times; i++) {
		vmc->press(key_spec->combination->get_buttons());
		timer.waitFor(std::chrono::milliseconds(interval));
	}
}

void VisitorInterpreter::visit_plug(const IR::Plug& plug) {
	try {
		if (plug.entity_type() == "nic") {
			return visit_plug_nic(plug);
		} else if (plug.entity_type() == "link") {
			return visit_plug_link(plug);
		} else if (plug.entity_type() == "dvd") {
			return visit_plug_dvd(plug);
		} else if (plug.entity_type() == "flash") {
			if(plug.is_on()) {
				return visit_plug_flash(plug);
			} else {
				return visit_unplug_flash(plug);
			}
		} else {
			throw std::runtime_error(std::string("unknown hardware type to plug/unplug: ") +
				plug.entity_type());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(plug.ast_node, current_controller));
	}
}

void VisitorInterpreter::visit_plug_nic(const IR::Plug& plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vmc while semantic analisys
	auto nic = plug.entity_name();

	auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
	reporter.plug(vmc, "nic", nic, plug.is_on());

	auto nics = vmc->vm()->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("specified nic {} is not present in this virtual machine", nic));
	}

	if (vmc->vm()->state() != VmState::Stopped) {
		throw std::runtime_error(fmt::format("virtual machine is running, but must be stopped"));
	}

	if (vmc->is_nic_plugged(nic) == plug.is_on()) {
		if (plug.is_on()) {
			throw std::runtime_error(fmt::format("specified nic {} is already plugged in this virtual machine", nic));
		} else {
			throw std::runtime_error(fmt::format("specified nic {} is not unplugged from this virtual machine", nic));
		}
	}

	if (plug.is_on()) {
		vmc->plug_nic(nic);
	} else {
		vmc->unplug_nic(nic);
	}
}

void VisitorInterpreter::visit_plug_link(const IR::Plug& plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vmc while semantic analisys

	auto nic = plug.entity_name();

	auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
	reporter.plug(vmc, "link", nic, plug.is_on());

	auto nics = vmc->vm()->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("the nic for specified link {} is not present in this virtual machine", nic));
	}

	if (!vmc->is_nic_plugged(nic)) {
		throw std::runtime_error(fmt::format("the nic for specified link {} is unplugged, you must to plug it first", nic));
	}

	if (plug.is_on() == vmc->is_link_plugged(nic)) {
		if (plug.is_on()) {
			throw std::runtime_error(fmt::format("specified link {} is already plugged in this virtual machine", nic));
		} else {
			throw std::runtime_error(fmt::format("specified link {} is already unplugged from this virtual machine", nic));
		}
	}

	if (plug.is_on()) {
		vmc->plug_link(nic);
	} else {
		vmc->unplug_link(nic);
	}
}

void VisitorInterpreter::visit_plug_flash(const IR::Plug& plug) {
	auto fdc = IR::program->get_flash_drive_or_throw(plug.entity_name());
	auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);

	reporter.plug(vmc, "flash drive", fdc->name(), true);
	for (auto vmc: current_test->get_all_machines()) {
		if (vmc->vm()->is_flash_plugged(fdc->fd())) {
			throw std::runtime_error(fmt::format("Flash drive {} is already plugged into vm {}. You should unplug it first", fdc->name(), vmc->name()));
		}
	}

	for (auto fdc: current_test->get_all_flash_drives()) {
		if (vmc->vm()->is_flash_plugged(fdc->fd())) {
			throw std::runtime_error(fmt::format("Another flash drive {} is already plugged into vm {}. You should unplug it first", fdc->name(), vmc->name()));
		}
	}

	vmc->vm()->plug_flash_drive(fdc->fd());
}

void VisitorInterpreter::visit_unplug_flash(const IR::Plug& plug) {
	auto fdc = IR::program->get_flash_drive_or_throw(plug.entity_name());
	auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);

	reporter.plug(vmc, "flash drive", fdc->name(), false);
	if (!vmc->vm()->is_flash_plugged(fdc->fd())) {
		throw std::runtime_error(fmt::format("specified flash {} is already unplugged from this virtual machine", fdc->name()));
	}

	vmc->vm()->unplug_flash_drive(fdc->fd());
}

void VisitorInterpreter::visit_plug_dvd(const IR::Plug& plug) {
	auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
	if (plug.is_on()) {
		auto path = plug.dvd_path();
		reporter.plug(vmc, "dvd", path.generic_string(), true);

		if (vmc->vm()->is_dvd_plugged()) {
			throw std::runtime_error(fmt::format("some dvd is already plugged"));
		}
		vmc->vm()->plug_dvd(path);
	} else {
		reporter.plug(vmc, "dvd", "", false);

		if (!vmc->vm()->is_dvd_plugged()) {
			std::cout << "DVD is already unplugged" << std::endl;
			// не считаем ошибкой, потому что дисковод мог быть вынут программным образом
			return;
		}
		vmc->vm()->unplug_dvd();

		auto deadline = std::chrono::system_clock::now() +  std::chrono::seconds(10);
		while (std::chrono::system_clock::now() < deadline) {
			if (!vmc->vm()->is_dvd_plugged()) {
				return;
			}
			timer.waitFor(std::chrono::milliseconds(300));
		}

		throw std::runtime_error(fmt::format("Timeout expired for unplugging dvd"));
	}
}

void VisitorInterpreter::visit_start(const IR::Start& start) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.start(vmc);
		vmc->vm()->start();
		auto deadline = std::chrono::system_clock::now() +  std::chrono::milliseconds(5000);
		while (std::chrono::system_clock::now() < deadline) {
			if (vmc->vm()->state() == VmState::Running) {
				return;
			}
			timer.waitFor(std::chrono::milliseconds(300));
		}
		throw std::runtime_error("Start timeout");
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(start.ast_node, current_controller));
	}
}

void VisitorInterpreter::visit_stop(const IR::Stop& stop) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.stop(vmc);
		vmc->vm()->stop();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(stop.ast_node, current_controller));

	}
}

void VisitorInterpreter::visit_shutdown(const IR::Shutdown& shutdown) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		std::string wait_for = shutdown.timeout();
		reporter.shutdown(vmc, wait_for);
		vmc->vm()->power_button();
		auto deadline = std::chrono::system_clock::now() +  std::chrono::milliseconds(time_to_milliseconds(wait_for));
		while (std::chrono::system_clock::now() < deadline) {
			if (vmc->vm()->state() == VmState::Stopped) {
				return;
			}
			timer.waitFor(std::chrono::milliseconds(300));
		}
		throw std::runtime_error("Shutdown timeout");
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(shutdown.ast_node, current_controller));

	}
}

static std::string build_shell_script(const std::string& body) {
	std::string script = "set -e; set -o pipefail; set -x;";
	script += body;
	script.erase(std::remove(script.begin(), script.end(), '\r'), script.end());

	return script;
}

static std::string build_batch_script(const std::string& body) {
	std::string script = "chcp 65001\n";
	script += body;
	return script;
}

static std::string build_python_script(const std::string& body) {
	std::vector<std::string> strings;
	std::stringstream iss(body);
	while(iss.good())
	{
		std::string single_string;
		getline(iss,single_string,'\n');
		strings.push_back(single_string);
	}

	size_t base_offset = 0;

	bool offset_found = false;

	for (auto& str: strings) {
		size_t offset_probe = 0;

		if (offset_found) {
			break;
		}

		for (auto it = str.begin(); it != str.end(); ++it) {
			while (*it == '\t') {
				offset_probe++;
				++it;
			}
			if (it == str.end()) {
				//empty string
				break;
			} else {
				//meaningful_string
				base_offset = offset_probe;
				offset_found = true;
				break;
			}
		}
	}

	std::string result;

	for (auto& str: strings) {
		for (auto it = str.begin(); it != str.end(); ++it) {
			for (size_t i = 0; i < base_offset; i++) {
				if (it == str.end()) {
					break;
				}

				if (*it != '\t') {
					throw std::runtime_error("Ill-formatted python script");
				}
				++it;
			}

			result += std::string(it, str.end());
			result += "\n";
			break;
		}
	}

	return result;
}


void VisitorInterpreter::visit_exec(const IR::Exec& exec) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.exec(vmc, exec.interpreter(), exec.timeout());

		if (vmc->vm()->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		if (!vmc->vm()->is_additions_installed()) {
			throw std::runtime_error(fmt::format("guest additions is not installed"));
		}

		std::string script, extension, interpreter;
		std::vector<std::string> args;

		if (exec.interpreter() == "bash") {
			script = build_shell_script(exec.script());
			extension = ".sh";
			interpreter = "bash";
		} else if (exec.interpreter() == "cmd") {
			script = build_batch_script(exec.script());
			extension = ".bat";
			interpreter = "cmd";
			args.push_back("/c");
		} else if (exec.interpreter() == "python") {
			script = build_python_script(exec.script());
			extension = ".py";
			interpreter = "python";
		} else if (exec.interpreter() == "python2") {
			script = build_python_script(exec.script());
			extension = ".py";
			interpreter = "python2";
		} else {
			script = build_python_script(exec.script());
			extension = ".py";
			interpreter = "python3";
		}

		//copy the script to tmp folder
		std::hash<std::string> h;

		std::string hash = std::to_string(h(script));

		fs::path host_script_dir = fs::temp_directory_path();
		fs::path guest_script_dir = vmc->vm()->get_tmp_dir();

		fs::path host_script_file = host_script_dir / std::string(hash + extension);
		fs::path guest_script_file = guest_script_dir / std::string(hash + extension);
		std::ofstream script_stream(host_script_file, std::ios::binary);
		if (!script_stream.is_open()) {
			throw std::runtime_error(fmt::format("Can't open tmp file for writing the script"));
		}

		script_stream << script;
		script_stream.close();

		vmc->vm()->copy_to_guest(host_script_file, guest_script_file); //5 seconds should be enough to pass any script

		fs::remove(host_script_file.generic_string());

		args.push_back(guest_script_file.generic_string());

		coro::Timeout timeout(std::chrono::milliseconds(time_to_milliseconds(exec.timeout())));

		auto result = vmc->vm()->run(interpreter, args, [&](const std::string& output) {
			reporter.exec_command_output(output);
		});
		if (result != 0) {
			throw std::runtime_error(interpreter + " command failed");
		}
		vmc->vm()->remove_from_guest(guest_script_file);

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(exec.ast_node, current_controller));
	}
}

void VisitorInterpreter::visit_copy(const IR::Copy& copy) {
	try {
		fs::path from = copy.from();
		fs::path to = copy.to();

		std::string wait_for = copy.timeout();
		reporter.copy(current_controller, from.generic_string(), to.generic_string(), copy.ast_node->is_to_guest(), wait_for);

		coro::Timeout timeout(std::chrono::milliseconds(time_to_milliseconds(wait_for)));

		if (auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller)) {
			if (vmc->vm()->state() != VmState::Running) {
				throw std::runtime_error(fmt::format("virtual machine is not running"));
			}

			if (!vmc->vm()->is_additions_installed()) {
				throw std::runtime_error(fmt::format("guest additions are not installed"));
			}

			if(copy.ast_node->is_to_guest()) {
				vmc->vm()->copy_to_guest(from, to);
			} else {
				vmc->vm()->copy_from_guest(from, to);;
			}
		}
		else {
			auto fdc = std::dynamic_pointer_cast<IR::FlashDrive>(current_controller);

			for (auto vmc: current_test->get_all_machines()) {
				if (vmc->vm()->is_flash_plugged(fdc->fd())) {
					throw std::runtime_error(fmt::format("Flash drive {} is already plugged into vm {}. You should unplug it first", fdc->name(), vmc->name()));
				}
			}

			//TODO: timeouts
			if(copy.ast_node->is_to_guest()) {
				fdc->fd()->upload(from, to);
			} else {
				fdc->fd()->download(from, to);
			}

		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(copy.ast_node, current_controller));
	}

}

void VisitorInterpreter::visit_macro_call(std::shared_ptr<AST::MacroCall> macro_call) {
	std::vector<std::pair<std::string, std::string>> args;
	std::map<std::string, std::string> vars;
	auto macro = IR::program->get_macro_or_throw(macro_call->name().value());

	for (size_t i = 0; i < macro_call->args.size(); ++i) {
		auto value = template_parser.resolve(macro_call->args[i]->text(), stack);
		vars[macro->ast_node->args[i]->name()] = value;
		args.push_back(std::make_pair(macro->ast_node->args[i]->name(), value));
	}

	for (size_t i = macro_call->args.size(); i < macro->ast_node->args.size(); ++i) {
		auto value = template_parser.resolve(macro->ast_node->args[i]->default_value->text(), stack);
		vars[macro->ast_node->args[i]->name()] = value;
		args.push_back(std::make_pair(macro->ast_node->args[i]->name(), value));
	}

	auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
	reporter.macro_call(vmc, macro_call->name(), args);

	StackPusher<VisitorInterpreter> new_ctx(this, macro->new_stack(vars));
	try {
		visit_action_block(macro->ast_node->action_block->action);
	} catch (const std::exception& error) {
		std::throw_with_nested(MacroException(macro_call));
	}
}

void VisitorInterpreter::visit_if_clause(std::shared_ptr<AST::IfClause> if_clause) {
	bool expr_result;
	try {
		expr_result = visit_expr(if_clause->expr);
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(if_clause, current_controller));
	}
	//everything else should be caught at test level
	if (expr_result) {
		return visit_action(if_clause->if_action);
	} else if (if_clause->has_else()) {
		return visit_action(if_clause->else_action);
	}
}

std::vector<std::string> VisitorInterpreter::visit_range(const IR::Range& range) {
	return range.values();
}

void VisitorInterpreter::visit_for_clause(std::shared_ptr<AST::ForClause> for_clause) {
	uint32_t i = 0;

	std::vector<std::string> values;

	if (auto p = std::dynamic_pointer_cast<AST::CounterList<AST::Range>>(for_clause->counter_list)) {
		values = visit_range({p->counter_list, stack});
	} else {
		throw std::runtime_error("Unknown counter list type");
	}

	std::map<std::string, std::string> vars;
	for (i = 0; i < values.size(); ++i) {
		vars[for_clause->counter.value()] = values[i];

		try {
			auto new_stack = std::make_shared<StackNode>();
			new_stack->parent = stack;
			new_stack->vars = vars;
			StackPusher<VisitorInterpreter> new_ctx(this, new_stack);
				visit_action(for_clause->cycle_body);

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

	if ((i == values.size()) && for_clause->else_token) {
		visit_action(for_clause->else_action);
	}
}

bool VisitorInterpreter::visit_expr(std::shared_ptr<AST::IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::BinOp>>(expr)) {
		return visit_binop(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::IFactor>>(expr)) {
		return visit_factor(p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

bool VisitorInterpreter::visit_binop(std::shared_ptr<AST::BinOp> binop) {
	auto left = visit_expr(binop->left);

	if (binop->op().value() == "AND") {
		if (!left) {
			return left;
		} else {
			return visit_expr(binop->right);
		}
	} else if (binop->op().value() == "OR") {
		if (left) {
			return left;
		} else {
			return visit_expr(binop->right);
		}
	} else {
		throw std::runtime_error("Unknown binop operation");
	}
}

bool VisitorInterpreter::visit_factor(std::shared_ptr<AST::IFactor> factor) {
	bool is_negated = factor->is_negated();

	if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::String>>(factor)) {
		return is_negated ^ (bool)template_parser.resolve(p->factor->text(), stack).length();
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Comparison>>(factor)) {
		return is_negated ^ visit_comparison({p->factor, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Defined>>(factor)) {
		return is_negated ^ visit_defined({p->factor, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Check>>(factor)) {
		return is_negated ^ visit_check({p->factor, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::ParentedExpr>>(factor)) {
		return is_negated ^ visit_expr(p->factor->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::IExpr>>(factor)) {
		return is_negated ^ visit_expr(p->factor);
	} else {
		throw std::runtime_error("Unknown factor type");
	}
}

bool VisitorInterpreter::visit_comparison(const IR::Comparison& comparison) {
	return comparison.calculate();
}

bool VisitorInterpreter::visit_defined(const IR::Defined& defined) {
	return defined.is_defined();
}

bool VisitorInterpreter::visit_check(const IR::Check& check) {
	try {
		std::string check_for = check.timeout();
		std::string interval_str = check.interval();
		auto interval = std::chrono::milliseconds(time_to_milliseconds(interval_str));
		auto text = template_parser.resolve(std::string(*check.ast_node->select_expr), check.stack);

		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.check(vmc, text, check_for, interval_str);

		auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(time_to_milliseconds(check_for));

		while (std::chrono::system_clock::now() < deadline) {
			auto start = std::chrono::high_resolution_clock::now();
			auto screenshot = vmc->vm()->screenshot();

			if (visit_detect_expr(check.ast_node->select_expr, screenshot)) {
				return true;
			}

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time = end - start;
			//std::cout << "time = " << time.count() << " seconds" << std::endl;
			if (interval > end - start) {
				timer.waitFor(interval - (end - start));
			} else {
				coro::CheckPoint();
			}
		}

		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(check.ast_node, current_controller));
	}
}

void VisitorInterpreter::stop_all_vms(std::shared_ptr<IR::Test> test) {
	for (auto vmc: test->get_all_machines()) {
		if (vmc->is_defined()) {
			if (vmc->vm()->state() != VmState::Stopped) {
				vmc->vm()->stop();
			}
			vmc->current_state = "";
		}
	}
}

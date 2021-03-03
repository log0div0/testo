
#include <coro/CheckPoint.h>
#include <coro/AsioTask.h>
#include "VisitorInterpreter.hpp"
#include "VisitorInterpreterActionMachine.hpp"
#include "VisitorInterpreterActionFlashDrive.hpp"
#include "IR/Program.hpp"
#include "Exceptions.hpp"
#include "Parser.hpp"

#include <fmt/format.h>
#include <wildcards.hpp>

void VisitorInterpreterConfig::validate() const {
	ReporterConfig::validate();
}

VisitorInterpreter::VisitorInterpreter(const VisitorInterpreterConfig& config) {
	reporter = Reporter(config);

	stop_on_fail = config.stop_on_fail;
	assume_yes = config.assume_yes;
	invalidate = config.invalidate;
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
				return true;
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
				bool is_already_pushed = false;

				for (auto test: cache_missed_tests) {
					if (test == *test_it) {
						is_already_pushed = true;
						break;
					}
				}
				if (!is_already_pushed) {
					cache_missed_tests.push_back(*test_it);
				}
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
		std::cout << "Because of the cache loss, Testo is scheduled to run the following tests:" << std::endl;

		for (auto cache_missed: cache_missed_tests) {
			std::cout << "\t- " << cache_missed->name() << std::endl;
		}

		std::cout << "Do you confirm running them? [y/N]: ";
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
		throw TestFailedException();
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

	} catch (const ControllerCreatonException& error) {
		std::stringstream ss;
		ss << error << std::endl;
		reporter.test_failed(ss.str());

		if (stop_on_fail) {
			throw std::runtime_error("");
		}

		stop_all_vms(test);
	} catch (const Exception& error) {
		std::stringstream ss;
		for (auto macro_call: test->macro_call_stack) {
			ss << std::string(macro_call->begin()) + std::string(": In a macro call ") << macro_call->name().value() << std::endl;
		}

		ss << error << std::endl;

		if (current_controller) {
			ss << std::endl;
			for (auto macro_call: current_controller->macro_call_stack) {
				ss << std::string(macro_call->begin()) + std::string(": In a macro call ") << macro_call->name().value() << std::endl;
			}
			ss << std::string(current_controller->ast_node->begin()) << ": note: the " << current_controller->type() << " " << current_controller->name() << " was declared here\n\n";
		}

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

void VisitorInterpreter::visit_command(std::shared_ptr<AST::ICmd> cmd) {
	if (auto p = std::dynamic_pointer_cast<AST::Cmd<AST::RegularCmd>>(cmd)) {
		visit_regular_command({p->cmd, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Cmd<AST::MacroCall>>(cmd)) {
		visit_macro_call({p->cmd, stack});
	} else {
		throw std::runtime_error("Should never happen");
	}
}

void VisitorInterpreter::visit_regular_command(const IR::RegularCommand& regular_command) {
	if (auto current_controller = IR::program->get_machine_or_null(regular_command.entity())) {
		this->current_controller = current_controller;
		VisitorInterpreterActionMachine(current_controller, stack, reporter, current_test).visit_action(regular_command.ast_node->action);
		this->current_controller = nullptr;
	} else if (auto current_controller = IR::program->get_flash_drive_or_null(regular_command.entity())) {
		this->current_controller = current_controller;
		VisitorInterpreterActionFlashDrive(current_controller, stack, reporter, current_test).visit_action(regular_command.ast_node->action);
		this->current_controller = nullptr;
	} else {
		throw std::runtime_error("Should never happen");
	}
}

void VisitorInterpreter::visit_macro_call(const IR::MacroCall& macro_call) {
	reporter.macro_command_call(macro_call.ast_node->name(), macro_call.args());
	macro_call.visit_interpreter<AST::MacroBodyCommand>(this);
}

void VisitorInterpreter::visit_macro_body(const std::shared_ptr<AST::MacroBodyCommand>& macro_body) {
	visit_command_block(macro_body->cmd_block);
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


#include <coro/CheckPoint.h>
#include <coro/AsioTask.h>
#include "VisitorInterpreter.hpp"
#include "VisitorInterpreterActionMachine.hpp"
#include "VisitorInterpreterActionFlashDrive.hpp"
#include "../IR/Program.hpp"
#include "../Exceptions.hpp"
#include "../Logger.hpp"

#include <fmt/format.h>
#include <wildcards.hpp>

VisitorInterpreter::VisitorInterpreter(const VisitorInterpreterConfig& config): reporter(config) {
	TRACE();

	stop_on_fail = config.stop_on_fail;
	assume_yes = config.assume_yes;
	invalidate = config.invalidate;
	dry = config.dry;
}

VisitorInterpreter::~VisitorInterpreter() {
	TRACE();
}

void VisitorInterpreter::delete_snapshot_with_children(const std::shared_ptr<IR::Test>& test) {
	for (auto& controller: test->get_all_controllers()) {
		if (controller->is_defined() && controller->has_snapshot(test->name())) {
			controller->delete_snapshot_with_children(test->name());
		}
	}
}

void VisitorInterpreter::invalidate_tests() {
	TRACE();

	if (!invalidate.length()) {
		return;
	}
	for (auto& test: IR::program->all_selected_tests) {
		if (wildcards::match(test->name(), invalidate)) {
			delete_snapshot_with_children(test);
		}
	}
}

void VisitorInterpreter::check_cache_missed_tests() {
	TRACE();

	if (assume_yes) {
		return;
	}

	std::vector<std::shared_ptr<IR::Test>> cache_missed_tests;

	for (auto& test: IR::program->all_selected_tests) {
		if (test->cache_status() == IR::Test::CacheStatus::Miss) {
			cache_missed_tests.push_back(test);
		}
	}

	if (!cache_missed_tests.size()) {
		return;
	}

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

std::shared_ptr<IR::TestRun> VisitorInterpreter::add_test_to_plan(const std::shared_ptr<IR::Test>& test) {
	// если тест up-to-date и есть снепшот - то запускать его точно не надо
	if (test->is_up_to_date() && test->has_hypervisor_snapshot()) {
		return nullptr;
	}
	// попробуем определить, может быть тест недавно выполнялся
	// и все контроллеры уже в нужном состоянии
	auto it = tests_runs.rbegin();
	auto controllers = test->get_all_controllers();
	for (; it != tests_runs.rend(); ++it) {
		auto test_run = *it;
		if (test_run->test == test) {
			return test_run;
		}
		auto other_controllers = test_run->test->get_all_controllers();
		if (std::find_first_of(
				controllers.begin(), controllers.end(),
				other_controllers.begin(), other_controllers.end()
			) != controllers.end()
		) {
			break;
		}
	}
	// контроллеры точно не в нужном состоянии, но
	// если тест уже был запланирован, и тест создаст снепшоты,
	// то мы сможем восстановить состояние контроллеров
	if (test->snapshot_policy() != IR::Test::SnapshotPolicy::Never) {
		for (; it != tests_runs.rend(); ++it) {
			auto test_run = *it;
			if (test_run->test == test) {
				return test_run;
			}
		}
	}
	// всё-таки придётся запланировать тест
	auto test_run = std::make_shared<IR::TestRun>();
	test_run->test = test;
	for (auto& parent_test: test->parents) {
		auto parent_test_run = add_test_to_plan(parent_test);
		if (parent_test_run) {
			if (parent_test_run != tests_runs.back()) {
				// дополнительный прогон родительского теста не был добавлен в список
				// вместо этого мы решили восстановить состояние родительских контроллеров
				// из снепшотов
				parent_test_run->test->add_snapshot_ref(test_run.get());
			}
			test_run->parents.insert(parent_test_run);
		} else {
			// the parent test is up-to-date
			parent_test->add_snapshot_ref(test_run.get());
		}
	}
	tests_runs.push_back(test_run);
	return test_run;
}

std::list<std::shared_ptr<IR::Test>> VisitorInterpreter::get_topmost_uncached_tests() {
	std::list<std::shared_ptr<IR::Test>> root_tests;
	for (auto& test: IR::program->all_selected_tests) {
		if (test->parents.size() == 0) {
			root_tests.push_back(test);
		}
	}

	std::list<std::shared_ptr<IR::Test>> result;
	std::set<std::shared_ptr<IR::Test>> visited_tests;
	std::stack<std::list<std::shared_ptr<IR::Test>>> stack;
	stack.push(std::move(root_tests));
	while (stack.size()) {
		if (stack.top().size() == 0) {
			stack.pop();
		} else {
			std::shared_ptr<IR::Test> test = stack.top().front();
			stack.top().pop_front();
			if (test->all_parents_are_up_to_date()) {
				if (test->is_up_to_date()) {
					stack.push(test->get_children());
				} else {
					result.push_back(test);
				}
			}
		}
	}
	return result;
}

struct DFSStackEntry {
	DFSStackEntry(const std::shared_ptr<IR::Test>& test_, std::set<std::string>& visited_tests_)
		: visited_tests(visited_tests_), test(test_), children_to_visit(test->get_children())
	{
	}
	DFSStackEntry(const std::list<std::shared_ptr<IR::Test>>& children_, std::set<std::string>& visited_tests_)
		: visited_tests(visited_tests_), children_to_visit(children_)
	{
	}

	std::set<std::string>& visited_tests;
	std::shared_ptr<IR::Test> test;
	std::list<std::shared_ptr<IR::Test>> children_to_visit;

	bool is_leaf() const {
		return test->children.size() == 0;
	}

	bool is_finished() const {
		return children_to_visit.size() == 0;
	}

	DFSStackEntry next() {
		if (children_to_visit.size() == 0) {
			throw std::runtime_error("Internal error: Do not call 'next()' method if `is_finished()` returns false");
		}

		auto it = std::find_if(children_to_visit.begin(), children_to_visit.end(), [&](const std::shared_ptr<IR::Test>& test) {
			return get_unresolved_dependencies(test).size() == 0;
		});

		if (it == children_to_visit.end()) {
			throw_unresolved_error();
		}

		std::shared_ptr<IR::Test> next_test = *it;
		children_to_visit.erase(it);
		visited_tests.insert(next_test->name());

		return { next_test, visited_tests };
	}

private:
	std::vector<std::string> get_unresolved_dependencies(const std::shared_ptr<IR::Test>& test) const {
		std::vector<std::string> result;
		for (const std::string& dependency: test->get_external_dependencies()) {
			if (!visited_tests.count(dependency)) {
				result.push_back(dependency);
			}
		}
		return result;
	}
	void throw_unresolved_error() const {
		std::string error_msg = "Can't decide which test to execute first because they depens on each other: ";
		int i = 0;
		for (auto test: children_to_visit) {
			if (i++) {
				error_msg += ", ";
			}
			error_msg += test->name();
		}
		throw std::runtime_error(error_msg);
	}
};

std::vector<std::shared_ptr<IR::Test>> VisitorInterpreter::get_leaf_tests_in_dfs_order(const std::list<std::shared_ptr<IR::Test>>& topmost_uncached_tests) {
	std::set<std::string> visited_tests;
	for (auto& test: IR::program->all_selected_tests) {
		if (test->is_up_to_date()) {
			visited_tests.insert(test->name());
		}
	}

	std::vector<std::shared_ptr<IR::Test>> result;
	std::stack<DFSStackEntry> stack;
	stack.push({topmost_uncached_tests, visited_tests});
	while (stack.size()) {
		if (stack.top().is_finished()) {
			if (stack.top().test && stack.top().is_leaf()) {
				// in case of multiple inheritance we can reach a leaf test
				// by multiple ways, so need the next check to avoid dublicates in
				// the resulting array
				if (std::find(result.begin(), result.end(), stack.top().test) == result.end()) {
					result.push_back(stack.top().test);
				}
			}
			stack.pop();
		} else {
			stack.push({stack.top().next()});
		}
	}
	return result;
}

void VisitorInterpreter::build_test_plan() {
	TRACE();

	std::list<std::shared_ptr<IR::Test>> topmost_uncached_tests = get_topmost_uncached_tests();
	for (auto& test: topmost_uncached_tests) {
		delete_snapshot_with_children(test);
	}
	std::vector<std::shared_ptr<IR::Test>> leaf_tests_in_dfs_order = get_leaf_tests_in_dfs_order(topmost_uncached_tests);
	for (auto& test: leaf_tests_in_dfs_order) {
		add_test_to_plan(test);
	}
}

void VisitorInterpreter::init() {
	TRACE();

	invalidate_tests();
	check_cache_missed_tests();
	build_test_plan();
}

void VisitorInterpreter::visit() {
	TRACE();

	init();

	if (dry) {
		return;
	}

	reporter.init(IR::program->all_selected_tests, tests_runs);

	for (size_t current_test_run_index = 0; current_test_run_index < tests_runs.size(); ++current_test_run_index) {
		auto test_run = tests_runs.at(current_test_run_index);

		//Check if one of the parents failed. If it did, just fail
		bool skip_test = false;

		for (auto parent: test_run->parents) {
			if (parent->exec_status != IR::TestRun::ExecStatus::Passed) {
				skip_test = true;
			}
		}
		for (auto parent: test_run->test->parents) {
			parent->remove_snapshot_ref(test_run.get());
		}

		for (auto dep: test_run->test->depends_on()) {
			// "depends_on" attribute is used to order tests, that have side effects
			// so we are interested only in the last test run of our dependency
			// (so side effects are "fresh")
			for (int64_t i = int64_t(current_test_run_index) - 1; i >= 0; --i) {
				auto finished_test_run = tests_runs.at(i);
				if (finished_test_run->test->name() == dep) {
					if (finished_test_run->exec_status != IR::TestRun::ExecStatus::Passed) {
						test_run->unsuccessful_deps_names.insert(dep);
						skip_test = true;
					}
					break;
				}
			}
		}

		if (skip_test) {
			delete_parents_hypervisor_snapshots_if_needed(test_run->test);
			reporter.skip_test();
			continue;
		}

		visit_test(test_run->test);

		//We need to check if we need to stop all the vms
		//VMS should be stopped if we don't need them anymore
		//and this could happen only if there's no children tests
		//ahead

		bool need_to_stop = true;

		if (test_run->exec_status == IR::TestRun::ExecStatus::Passed) {
			for (size_t i = current_test_run_index; i < tests_runs.size(); ++i) {
				for (auto parent: tests_runs.at(i)->parents) {
					if (parent == test_run) {
						need_to_stop = false;
						break;
					}
				}
				if (!need_to_stop) {
					break;
				}
			}
		}

		if (need_to_stop) {
			stop_all_vms(test_run->test);
		}
	}

	reporter.finish();
	if (reporter.get_stats(IR::TestRun::ExecStatus::Failed).size()) {
		throw TestFailedException();
	}
}

void VisitorInterpreter::delete_parents_hypervisor_snapshots_if_needed(const std::shared_ptr<IR::Test>& test) {
	for (auto parent: test->parents) {
		if (parent->can_delete_hypervisor_snaphots()) {
			for (auto controller: parent->get_all_controllers()) {
				if (controller->has_hypervisor_snapshot(parent->name())) {
					reporter.delete_hypervisor_snapshot(controller, parent->name());
					controller->delete_hypervisor_snapshot(parent->name());
					coro::CheckPoint();
				}
			}
		}
	}
}

void VisitorInterpreter::restore_parents_controllers_if_needed(const std::shared_ptr<IR::Test>& test) {
	for (auto parent: test->parents) {
		for (auto controller: parent->get_all_controllers()) {
			if (controller->current_state != parent->name()) {
				reporter.restore_snapshot(controller, parent->name());
				controller->restore_snapshot(parent->name());
				coro::CheckPoint();
			}
		}
	}
}

void VisitorInterpreter::create_networks_if_needed(const std::shared_ptr<IR::Test>& test) {
	for (auto netc: test->get_all_networks()) {
		if (netc->is_defined() &&
			netc->check_config_relevance())
		{
			continue;
		}
		netc->create();
	}
}

void VisitorInterpreter::install_new_controllers_if_needed(const std::shared_ptr<IR::Test>& test) {
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
}

void VisitorInterpreter::resume_parents_vms(const std::shared_ptr<IR::Test>& test) {
	for (auto parent: test->parents) {
		for (auto vmc: parent->get_all_machines()) {
			if (vmc->vm()->state() == VmState::Suspended) {
				vmc->vm()->resume();
			}
		}
	}
}

void VisitorInterpreter::suspend_all_vms(const std::shared_ptr<IR::Test>& test) {
	for (auto vmc: test->get_all_machines()) {
		if (vmc->vm()->state() == VmState::Running) {
			vmc->vm()->suspend();
		}
	}
}

void VisitorInterpreter::create_all_controllers_snapshots(const std::shared_ptr<IR::Test>& test) {
	//we need to take snapshots in the right order
	//1) all the vms - so we could check that all the fds are unplugged
	for (auto controller: test->get_all_machines()) {
		if (!controller->has_snapshot(test->name(), test->is_hypervisor_snapshot_needed())) {
			reporter.take_snapshot(controller, test->name());
			controller->create_snapshot(test->name(), test->cksum, test->is_hypervisor_snapshot_needed());
			coro::CheckPoint();
		}
		controller->current_state = test->name();
	}

	//2) all the fdcs - the rest
	for (auto controller: test->get_all_flash_drives()) {
		if (!controller->has_snapshot(test->name(), test->is_hypervisor_snapshot_needed())) {
			reporter.take_snapshot(controller, test->name());
			controller->create_snapshot(test->name(), test->cksum, test->is_hypervisor_snapshot_needed());
			coro::CheckPoint();
		}
		controller->current_state = test->name();
	}
}

void VisitorInterpreter::visit_test(const std::shared_ptr<IR::Test>& test) {
	try {
		current_test = nullptr;

		reporter.prepare_environment();

		restore_parents_controllers_if_needed(test);
		create_networks_if_needed(test);
		install_new_controllers_if_needed(test);

		resume_parents_vms(test);
		{
			reporter.run_test();
			StackPusher<VisitorInterpreter> pusher(this, test->stack);
			current_test = test;
			visit_command_block(test->ast_node->cmd_block);
		}
		suspend_all_vms(test);

		delete_parents_hypervisor_snapshots_if_needed(test);
		create_all_controllers_snapshots(test);

		reporter.test_passed();

	} catch (const Exception& error) {
		std::stringstream ss;
		ss << test->macro_call_stack << error << std::endl;

		if (current_controller) {
			ss << std::endl << current_controller->note_was_declared_here() << "\n\n";
		}

		std::string failure_category = GetFailureCategory(error);

		reporter.test_failed(error.what(), ss.str(), failure_category);

		if (stop_on_fail) {
			throw std::runtime_error("");
		}
	}
}

void VisitorInterpreter::visit_command_block(const std::shared_ptr<AST::Block<AST::Cmd>>& block) {
	for (auto command: block->items) {
		visit_command(command);
	}
}

void VisitorInterpreter::visit_command(const std::shared_ptr<AST::Cmd>& cmd) {
	if (auto p = std::dynamic_pointer_cast<AST::RegularCmd>(cmd)) {
		visit_regular_command({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MacroCall<AST::Cmd>>(cmd)) {
		visit_macro_call({p, stack});
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
	reporter.macro_command_call(macro_call);
	macro_call.visit_interpreter<AST::Cmd>(this);
}

void VisitorInterpreter::visit_macro_body(const std::shared_ptr<AST::Block<AST::Cmd>>& macro_body) {
	visit_command_block(macro_body);
}

void VisitorInterpreter::stop_all_vms(const std::shared_ptr<IR::Test>& test) {
	TRACE();

	for (auto vmc: test->get_all_machines()) {
		if (vmc->is_defined()) {
			if (vmc->vm()->state() != VmState::Stopped) {
				vmc->vm()->stop();
			}
			vmc->current_state = "";
		}
	}
}

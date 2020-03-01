
#include "VisitorInterpreter.hpp"
#include "VisitorCksum.hpp"

#include "coro/Finally.h"
#include "coro/CheckPoint.h"
#include "utf8.hpp"
#include <fmt/format.h>
#include <fstream>
#include <thread>
#include <wildcards.hpp>
#include <rang.hpp>

#include "nn/Context.hpp"

using namespace std::chrono_literals;

Reporter reporter;

static void sleep(const std::string& interval) {
	coro::Timer timer;
	timer.waitFor(std::chrono::milliseconds(time_to_milliseconds(interval)));
}

VisitorInterpreter::VisitorInterpreter(Register& reg, const nlohmann::json& config): reg(reg) {
	js_runtime = quickjs::create_runtime();

	reporter = Reporter(config);

	stop_on_fail = config.at("stop_on_fail").get<bool>();
	assume_yes = config.at("assume_yes").get<bool>();
	test_spec = config.at("test_spec").get<std::string>();
	exclude = config.at("exclude").get<std::string>();
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

	//check networks aditionally
	for (auto netc: reg.get_all_netcs(test)) {
		if (netc->is_defined() &&
			netc->check_config_relevance())
		{
			continue;
		}
		return false;
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

bool VisitorInterpreter::is_cache_miss(std::shared_ptr<AST::Test> test) const {
	auto all_parents = reg.get_test_path(test);

	for (auto parent: all_parents) {
		for (auto cache_missed_test: cache_missed_tests) {
			if (parent == cache_missed_test) {
				return false;
			}
		}
	}

	//check networks aditionally
	for (auto netc: reg.get_all_netcs(test)) {
		if (netc->is_defined()) {
			if (!netc->check_config_relevance()) {
				return true;
			}
		}
	}

	for (auto controller: reg.get_all_controllers(test)) {
		if (controller->is_defined()) {
			if (controller->has_snapshot(test->name.value())) {
				if (controller->get_snapshot_cksum(test->name.value()) != test_cksum(test)) {
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

void VisitorInterpreter::check_up_to_date_tests(std::list<std::shared_ptr<AST::Test>>& tests_queue) {
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

void VisitorInterpreter::resolve_tests(const std::list<std::shared_ptr<AST::Test>>& tests_queue) {
	for (auto test: tests_queue) {
		for (auto controller: reg.get_all_controllers(test)) {
			if (controller->is_defined() && controller->has_snapshot(test->name.value())) {
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
					if (controller->is_defined() && controller->has_snapshot(test->name.value())) {
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

	if (!assume_yes && cache_missed_tests.size()) {
		std::cout << "Some tests have lost their cache:" << std::endl;

		for (auto cache_missed: cache_missed_tests) {
			std::cout << "\t- " << cache_missed->name.value() << std::endl;
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
		for (auto controller: reg.get_all_controllers(test)) {
			if (controller->is_defined()) {
				controller->set_metadata("current_state", "");
			}
		}
	}
}

void VisitorInterpreter::visit(std::shared_ptr<AST::Program> program) {
	setup_vars(program);

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

void VisitorInterpreter::visit_test(std::shared_ptr<AST::Test> test) {
	try {
		//Ok, we're not cached and we need to run the test
		reporter.prepare_environment();
		//Check if one of the parents failed. If it did, just fail
		for (auto parent: test->parents) {
			for (auto failed: reporter.failed_tests) {
				if (parent->name.value() == failed->name) {
					reporter.skip_failed_test(parent->name);
					return;
				}
			}
		}

		//we need to get all the vms in the correct state
		//vms from parents - rollback them to parents if we need to
		//We need to do it only if our current state is not the parent
		for (auto parent: test->parents) {
			for (auto controller: reg.get_all_controllers(parent)) {
				if (controller->get_metadata("current_state") != parent->name.value()) {
					reporter.restore_snapshot(controller, parent->name);
					controller->restore_snapshot(parent->name.value());
				}
			}
		}

		//check all the networks

		for (auto netc: reg.get_all_netcs(test)) {
			if (netc->is_defined() &&
				netc->check_config_relevance())
			{
				continue;
			}
			netc->create();
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
					reporter.restore_snapshot(controller, "initial");
					controller->restore_snapshot("_init");
				} else {
					reporter.create_controller(controller);
					controller->create();

					reporter.take_snapshot(controller, "initial");
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

		reporter.run_test();

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
				reporter.take_snapshot(controller, test->name);
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
		reporter.test_passed();

	} catch (const InterpreterException& error) {
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
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		return visit_sleep(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Press>>(action)) {
		return visit_press(vmc, p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Mouse>>(action)) {
		return visit_mouse(vmc, p->action);
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
	std::string message = template_parser.resolve(abort->message->text(), reg);
	throw AbortException(abort, vmc, message);
}

void VisitorInterpreter::visit_print(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Print> print_action) {
	try {
		std::string message = template_parser.resolve(print_action->message->text(), reg);
		reporter.print(vmc, message);
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(print_action, vmc));
	}
}

void VisitorInterpreter::visit_type(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Type> type) {
	try {
		std::string text = template_parser.resolve(type->text->text(), reg);
		if (text.size() == 0) {
			return;
		}

		reporter.type(vmc, text);

		for (auto c: utf8::split_to_chars(text)) {
			auto buttons = charmap.find(c);
			if (buttons == charmap.end()) {
				throw std::runtime_error("Unknown character to type");
			}
			vmc->vm->press(buttons->second);
			timer.waitFor(std::chrono::milliseconds(30));
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(type, vmc));
	}
}

bool VisitorInterpreter::visit_select_expr(std::shared_ptr<AST::ISelectExpr> select_expr, stb::Image& screenshot) {
	if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::ISelectable>>(select_expr)) {
		return visit_select_selectable(p->select_expr, screenshot).size();
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::SelectUnOp>>(select_expr)) {
		return visit_select_unop(p->select_expr, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::SelectBinOp>>(select_expr)) {
		return visit_select_binop(p->select_expr, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::SelectParentedExpr>>(select_expr)) {
		return visit_select_expr(p->select_expr->select_expr, screenshot);
	} else {
		throw std::runtime_error("Unknown select expression type");
	}
}

quickjs::Value VisitorInterpreter::eval_js(const std::string& script, stb::Image& screenshot) {
	try {
		auto js_ctx = js_runtime.create_context();
		js_ctx.register_nn_functions();
		nn::Context nn_ctx(&screenshot);
		js_ctx.set_opaque(&nn_ctx);
		return js_ctx.eval(script);
	} catch(const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Error while executing javascript selection"));
	}

}

std::vector<nn::Rect> VisitorInterpreter::visit_select_selectable(std::shared_ptr<AST::ISelectable> selectable, stb::Image& screenshot) {
	if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::String>>(selectable)) {
		auto text = template_parser.resolve(p->text(), reg);
		return nn::OCR(&screenshot).search(text);
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(selectable)) {
		auto script = template_parser.resolve(p->text(), reg);
		auto value = eval_js(script, screenshot);
		std::vector<nn::Rect> result;
		if (value.is_bool() && (bool)value) {
			result.push_back(nn::Rect());
		}
		return result;
	} else {
		throw std::runtime_error("Unknown selectable type");
	}
}

bool VisitorInterpreter::visit_select_unop(std::shared_ptr<AST::SelectUnOp> unop, stb::Image& screenshot) {
	if (unop->t.type() == Token::category::exclamation_mark) {
		return !visit_select_expr(unop->select_expr, screenshot);
	} else {
		throw std::runtime_error("Unknown unop operation");
	}
}

bool VisitorInterpreter::visit_select_binop(std::shared_ptr<AST::SelectBinOp> binop, stb::Image& screenshot) {
	auto left_value = visit_select_expr(binop->left, screenshot);
	if (binop->t.type() == Token::category::double_ampersand) {
		if (!left_value) {
			return false;
		} else {
			return left_value && visit_select_expr(binop->right, screenshot);
		}
	} else if (binop->t.type() == Token::category::double_vertical_bar) {
		if (left_value) {
			return true;
		} else {
			return left_value || visit_select_expr(binop->right, screenshot);
		}
	} else {
		throw std::runtime_error("Unknown binop operation");
	}
}

void VisitorInterpreter::visit_sleep(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Sleep> sleep) {
	reporter.sleep(vmc, sleep->timeout.value());
	::sleep(sleep->timeout.value());
}

void VisitorInterpreter::visit_wait(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Wait> wait) {
	try {
		std::string wait_for = wait->timeout ? wait->timeout.value() : "1m";
		std::string interval_str = wait->interval ? wait->interval.value() : "1s";
		auto interval = std::chrono::milliseconds(time_to_milliseconds(interval_str));
		auto text = template_parser.resolve(std::string(*wait->select_expr), reg);

		reporter.wait(vmc, text, wait_for, interval_str);

		auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(time_to_milliseconds(wait_for));

		while (std::chrono::system_clock::now() < deadline) {
			auto start = std::chrono::high_resolution_clock::now();
			auto screenshot = vmc->vm->screenshot();

			if (visit_select_expr(wait->select_expr, screenshot)) {
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
		throw std::runtime_error("Wait timeout");

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(wait, vmc));
	}
}

void VisitorInterpreter::visit_press(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Press> press) {
	try {
		std::string interval = press->interval ? press->interval.value() : "30ms";
		auto press_interval = time_to_milliseconds(interval);

		for (auto key_spec: press->keys) {
			visit_key_spec(vmc, key_spec, press_interval);
			timer.waitFor(std::chrono::milliseconds(press_interval));
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(press, vmc));
	}
}

void VisitorInterpreter::visit_mouse(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Mouse> mouse) {
	if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseMoveClick>>(mouse->event)) {
		return visit_mouse_move_click(vmc, p->event);
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseHold>>(mouse->event)) {
		return visit_mouse_hold(vmc, p->event);
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseRelease>>(mouse->event)) {
		return visit_mouse_release(vmc, p->event);
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseWheel>>(mouse->event)) {
		return visit_mouse_wheel(vmc, p->event);
	} else {
		throw std::runtime_error("Unknown mouse actions");
	}
}

void VisitorInterpreter::visit_mouse_hold(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MouseHold> mouse_hold) {
	try {
		reporter.mouse_hold(vmc, mouse_hold->button.value());
		if (mouse_hold->button.type() == Token::category::lbtn) {
			vmc->vm->mouse_press({MouseButton::Left});
			vmc->current_held_mouse_button = MouseButton::Left;
		} else if (mouse_hold->button.type() == Token::category::rbtn) {
			vmc->vm->mouse_press({MouseButton::Right});
			vmc->current_held_mouse_button = MouseButton::Right;
		} else if (mouse_hold->button.type() == Token::category::mbtn) {
			vmc->vm->mouse_press({MouseButton::Middle});
			vmc->current_held_mouse_button = MouseButton::Middle;
		} else {
			throw std::runtime_error("Unknown mouse button: " + mouse_hold->button.value());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_hold, vmc));
	}
}

void VisitorInterpreter::visit_mouse_release(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MouseRelease> mouse_release) {
	try {
		reporter.mouse_release(vmc);
		if (vmc->current_held_mouse_button == MouseButton::Left) {
			vmc->vm->mouse_release({MouseButton::Left});
		} else if (vmc->current_held_mouse_button == MouseButton::Right) {
			vmc->vm->mouse_release({MouseButton::Right});
		} else if (vmc->current_held_mouse_button == MouseButton::Middle) {
			vmc->vm->mouse_release({MouseButton::Middle});
		} else if (vmc->current_held_mouse_button == MouseButton::None) {
			throw std::runtime_error("No mouse button is pressed right now");
		} else {
			throw std::runtime_error("Unknown button to release");
		}

		vmc->current_held_mouse_button = MouseButton::None;
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_release, vmc));
	}
}

void VisitorInterpreter::visit_mouse_wheel(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MouseWheel> mouse_wheel) {
	try {
		reporter.mouse_wheel(vmc, mouse_wheel->direction.value());

		if (mouse_wheel->direction.value() == "up") {
			vmc->vm->mouse_press({MouseButton::WheelUp});
			vmc->vm->mouse_release({MouseButton::WheelUp});
		} else if (mouse_wheel->direction.value() == "down") {
			vmc->vm->mouse_press({MouseButton::WheelDown});
			vmc->vm->mouse_release({MouseButton::WheelDown});
		} else {
			throw std::runtime_error("Unknown wheel direction");
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_wheel, vmc));
	}
}

void VisitorInterpreter::visit_mouse_move_click(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MouseMoveClick> mouse_move_click) {
	try {
		std::string where_to_go = mouse_move_click->object ? mouse_move_click->object->text() : "";
		std::string wait_for = mouse_move_click->timeout_interval ? mouse_move_click->timeout_interval.value() : "5s";
		reporter.mouse_move_click(vmc, mouse_move_click->t.value(), where_to_go, wait_for);

		if (mouse_move_click->object) {
			if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseCoordinates>>(mouse_move_click->object)) {
				visit_mouse_move_coordinates(vmc, p->target);
			} else if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::ISelectable>>(mouse_move_click->object)) {
				visit_mouse_move_selectable(vmc, p->target, wait_for);
			} else {
				throw std::runtime_error("Unknown mouse move target");
			}
		}

		if (mouse_move_click->t.type() == Token::category::move) {
			return;
		}

		if (vmc->current_held_mouse_button != MouseButton::None) {
			throw std::runtime_error("Can't click anything with a held mouse button");
		}

		if (mouse_move_click->t.type() == Token::category::click || mouse_move_click->t.type() == Token::category::lclick) {
			vmc->vm->mouse_press({MouseButton::Left});
			vmc->vm->mouse_release({MouseButton::Left});
		} else if (mouse_move_click->t.type() == Token::category::rclick) {
			vmc->vm->mouse_press({MouseButton::Right});
			vmc->vm->mouse_release({MouseButton::Right});
		} else if (mouse_move_click->t.type() == Token::category::mclick) {
			vmc->vm->mouse_press({MouseButton::Middle});
			vmc->vm->mouse_release({MouseButton::Middle});
		} else if (mouse_move_click->t.type() == Token::category::dclick) {
			vmc->vm->mouse_press({MouseButton::Left});
			vmc->vm->mouse_release({MouseButton::Left});
			vmc->vm->mouse_press({MouseButton::Left});
			vmc->vm->mouse_release({MouseButton::Left});
		} else {
			throw std::runtime_error("Unsupported click type");
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_move_click, vmc));
	}
}

void VisitorInterpreter::visit_mouse_move_selectable(std::shared_ptr<VmController> vmc,
	std::shared_ptr<AST::ISelectable> selectable, const std::string& timeout)
{
	auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(time_to_milliseconds(timeout));

	std::vector<nn::Rect> found;
	while (std::chrono::system_clock::now() < deadline) {
		auto start = std::chrono::high_resolution_clock::now();
		auto screenshot = vmc->vm->screenshot();

		found = visit_select_selectable(selectable, screenshot);
		if (found.size()) {
			break;
		}

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end - start;
		if (time < 1s) {
			timer.waitFor(std::chrono::milliseconds(std::chrono::duration_cast<std::chrono::milliseconds>(1s - time)));
		} else {
			coro::CheckPoint();
		}
	}

	if (!found.size()) {
		throw std::runtime_error("Can't find entry to click: " + selectable->text());
	}

	if (found.size() > 1) {
		throw std::runtime_error("Too many occurences of entry to click: " + selectable->text());
	}

	vmc->vm->mouse_move_abs(found[0].center_x(), found[0].center_y());
}

void VisitorInterpreter::visit_mouse_move_coordinates(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MouseCoordinates> coordinates)
{
	auto dx = coordinates->dx.value();
	if ((dx[0] == '+') || (dx[0] == '-')) {
		vmc->vm->mouse_move_rel("x", std::stoi(dx));
	} else {
		vmc->vm->mouse_move_abs("x", std::stoul(dx));
	}

	auto dy = coordinates->dy.value();
	if ((dy[0] == '+') || (dy[0] == '-')) {
		vmc->vm->mouse_move_rel("y", std::stoi(dy));
	} else {
		vmc->vm->mouse_move_abs("y", std::stoul(dy));
	}
}

//void VisitorInterpreter::visit_mouse_event(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MouseEvent> mouse_event) {

		/*std::string where_to_go = mouse_event->is_move_needed() ? mouse_event->object->text() : "";
		std::string wait_for_report = mouse_event->time_interval ? mouse_event->time_interval.value() : "";
		reporter.mouse_event(vmc, mouse_event->event.value(), where_to_go, wait_for_report);
	*/

//}

void VisitorInterpreter::visit_key_spec(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::KeySpec> key_spec, uint32_t interval) {
	uint32_t times = key_spec->get_times();

	reporter.press_key(vmc, key_spec->get_buttons_str(), times);

	for (uint32_t i = 0; i < times; i++) {
		vmc->vm->press(key_spec->get_buttons());
		timer.waitFor(std::chrono::milliseconds(interval));
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

	reporter.plug(vmc, "nic", nic, plug->is_on());

	auto nics = vmc->vm->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("specified nic {} is not present in this virtual machine", nic));
	}

	if (vmc->vm->state() != VmState::Stopped) {
		throw std::runtime_error(fmt::format("virtual machine is running, but must be stopped"));
	}

	if (vmc->vm->is_nic_plugged(nic) == plug->is_on()) {
		if (plug->is_on()) {
			throw std::runtime_error(fmt::format("specified nic {} is already plugged in this virtual machine", nic));
		} else {
			throw std::runtime_error(fmt::format("specified nic {} is not unplugged from this virtual machine", nic));
		}
	}

	vmc->vm->set_nic(nic, plug->is_on());
}

void VisitorInterpreter::visit_plug_link(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vmc while semantic analisys

	auto nic = plug->name_token.value();

	reporter.plug(vmc, "link", nic, plug->is_on());

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

	vmc->vm->set_link(nic, plug->is_on());
}

void VisitorInterpreter::plug_flash(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
	auto fdc = reg.fdcs.find(plug->name_token.value())->second; //should always be found

	reporter.plug(vmc, "flash drive", fdc->name(), true);
	if (vmc->vm->is_flash_plugged(fdc->fd)) {
		throw std::runtime_error(fmt::format("specified flash {} is already plugged into this virtual machine", fdc->name()));
	}

	vmc->vm->plug_flash_drive(fdc->fd);
}

void VisitorInterpreter::unplug_flash(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
	auto fdc = reg.fdcs.find(plug->name_token.value())->second; //should always be found

	reporter.plug(vmc, "flash drive", fdc->name(), false);
	if (!vmc->vm->is_flash_plugged(fdc->fd)) {
		throw std::runtime_error(fmt::format("specified flash {} is already unplugged from this virtual machine", fdc->name()));
	}

	vmc->vm->unplug_flash_drive(fdc->fd);
}

void VisitorInterpreter::visit_plug_dvd(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Plug> plug) {
	if (plug->is_on()) {
		fs::path path = template_parser.resolve(plug->path->text(), reg);
		if (path.is_relative()) {
			path = plug->t.pos().file.parent_path() / path;
		}

		reporter.plug(vmc, "dvd", path.generic_string(), true);

		if (vmc->vm->is_dvd_plugged()) {
			throw std::runtime_error(fmt::format("some dvd is already plugged"));
		}
		vmc->vm->plug_dvd(path);
	} else {
		if (!vmc->vm->is_dvd_plugged()) {
			// throw std::runtime_error(fmt::format("dvd is already unplugged"));
			// это нормально, потому что поведение отличается от гипервизора к гипервизору
			// иногда у ОС получается открыть дисковод, иногда - нет
			return;
		}
		reporter.plug(vmc, "dvd", "", false);
		vmc->vm->unplug_dvd();
	}
}

void VisitorInterpreter::visit_start(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Start> start) {
	try {
		reporter.start(vmc);
		vmc->vm->start();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(start, vmc));
	}
}

void VisitorInterpreter::visit_stop(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Stop> stop) {
	try {
		reporter.stop(vmc);
		vmc->vm->stop();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(stop, vmc));

	}
}

void VisitorInterpreter::visit_shutdown(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Shutdown> shutdown) {
	try {
		std::string wait_for = shutdown->time_interval ? shutdown->time_interval.value() : "1m";
		reporter.shutdown(vmc, wait_for);
		vmc->vm->power_button();
		auto deadline = std::chrono::system_clock::now() +  std::chrono::milliseconds(time_to_milliseconds(wait_for));
		while (std::chrono::system_clock::now() < deadline) {
			if (vmc->vm->state() == VmState::Stopped) {
				return;
			}
			timer.waitFor(std::chrono::milliseconds(300));
		}
		throw std::runtime_error("Shutdown timeout");
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(shutdown, vmc));

	}
}

static std::string build_shell_script(const std::string& body) {
	std::string script = "set -e; set -o pipefail; set -x;";
	script += body;
	script.erase(std::remove(script.begin(), script.end(), '\r'), script.end());

	return script;
}

static std::string build_batch_script(const std::string& body) {
	return body;
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


void VisitorInterpreter::visit_exec(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Exec> exec) {
	try {
		std::string wait_for = exec->time_interval ? exec->time_interval.value() : "10m";
		reporter.exec(vmc, exec->process_token.value(), wait_for);

		if (vmc->vm->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		if (!vmc->vm->is_additions_installed()) {
			throw std::runtime_error(fmt::format("guest additions is not installed"));
		}

		std::string script, extension, interpreter;
		std::vector<std::string> args;

		if (exec->process_token.value() == "bash") {
			script = build_shell_script(template_parser.resolve(exec->commands->text(), reg));
			extension = ".sh";
			interpreter = "bash";
		} else if (exec->process_token.value() == "cmd") {
			script = build_batch_script(template_parser.resolve(exec->commands->text(), reg));
			extension = ".bat";
			interpreter = "cmd";
			args.push_back("/c");
		} else if (exec->process_token.value() == "python") {
			script = build_python_script(template_parser.resolve(exec->commands->text(), reg));
			extension = ".py";
			interpreter = "python";
		} else if (exec->process_token.value() == "python2") {
			script = build_python_script(template_parser.resolve(exec->commands->text(), reg));
			extension = ".py";
			interpreter = "python2";
		} else {
			script = build_python_script(template_parser.resolve(exec->commands->text(), reg));
			extension = ".py";
			interpreter = "python3";
		}

		//copy the script to tmp folder
		std::hash<std::string> h;

		std::string hash = std::to_string(h(script));

		fs::path host_script_dir = fs::temp_directory_path();
		fs::path guest_script_dir = vmc->vm->get_tmp_dir();

		fs::path host_script_file = host_script_dir / std::string(hash + extension);
		fs::path guest_script_file = guest_script_dir / std::string(hash + extension);
		std::ofstream script_stream(host_script_file, std::ios::binary);
		if (!script_stream.is_open()) {
			throw std::runtime_error(fmt::format("Can't open tmp file for writing the script"));
		}

		script_stream << script;
		script_stream.close();

		vmc->vm->copy_to_guest(host_script_file, guest_script_file, 5000); //5 seconds should be enough to pass any script

		fs::remove(host_script_file.generic_string());

		args.push_back(guest_script_file.generic_string());
		if (vmc->vm->run(interpreter, args, time_to_milliseconds(wait_for)) != 0) {
			throw std::runtime_error(interpreter + " command failed");
		}
		vmc->vm->remove_from_guest(guest_script_file);

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(exec, vmc));
	}
}

void VisitorInterpreter::visit_copy(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Copy> copy) {
	try {
		fs::path from = template_parser.resolve(copy->from->text(), reg);
		fs::path to = template_parser.resolve(copy->to->text(), reg);

		std::string wait_for = copy->time_interval ? copy->time_interval.value() : "10m";
		reporter.copy(vmc, from.generic_string(), to.generic_string(), copy->is_to_guest(), wait_for);

		if (vmc->vm->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		if (!vmc->vm->is_additions_installed()) {
			throw std::runtime_error(fmt::format("guest additions are not installed"));
		}

		if(copy->is_to_guest()) {
			if (from.is_relative()) {
				from = copy->t.pos().file.parent_path() / from;
			}
			vmc->vm->copy_to_guest(from, to, time_to_milliseconds(wait_for));
		} else {
			if (to.is_relative()) {
				to = copy->t.pos().file.parent_path() / to;
			}
			vmc->vm->copy_from_guest(from, to, time_to_milliseconds(wait_for));;
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(copy, vmc));
	}

}

void VisitorInterpreter::visit_macro_call(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::MacroCall> macro_call) {
	//push new ctx
	StackEntry new_ctx(true);

	std::vector<std::pair<std::string, std::string>> params;

	for (size_t i = 0; i < macro_call->params.size(); ++i) {
		auto value = template_parser.resolve(macro_call->params[i]->text(), reg);
		new_ctx.define(macro_call->macro->params[i].value(), value);
		params.push_back(std::make_pair(macro_call->macro->params[i].value(), value));
	}
	reg.local_vars.push_back(new_ctx);
	coro::Finally finally([&] {
		reg.local_vars.pop_back();
	});

	reporter.macro_call(vmc, macro_call->name(), params);
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
	reg.local_vars.push_back(new_ctx);
	size_t ctx_position = reg.local_vars.size() - 1;
	coro::Finally finally([&]{
		reg.local_vars.pop_back();
	});
	uint32_t i = 0;
	for (i = for_clause->start(); i <= for_clause->finish(); i++) {
		reg.local_vars[ctx_position].define(for_clause->counter.value(), std::to_string(i));
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

	if ((i == for_clause->finish() + 1) && for_clause->else_token) {
		visit_action(vmc, for_clause->else_action);
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
	if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::String>>(factor)) {
		return p->is_negated() ^ (bool)template_parser.resolve(p->factor->text(), reg).length();
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

bool VisitorInterpreter::visit_comparison(std::shared_ptr<VmController> vmc, std::shared_ptr<AST::Comparison> comparison) {
	auto left = template_parser.resolve(comparison->left->text(), reg);
	auto right = template_parser.resolve(comparison->right->text(), reg);
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
		std::string check_for = check->timeout ? check->timeout.value() : "1ms";
		std::string interval_str = check->interval ? check->interval.value() : "1s";
		auto interval = std::chrono::milliseconds(time_to_milliseconds(interval_str));
		auto text = template_parser.resolve(std::string(*check->select_expr), reg);
		reporter.check(vmc, text, check_for, interval_str);

		auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(time_to_milliseconds(check_for));

		while (std::chrono::system_clock::now() < deadline) {
			auto start = std::chrono::high_resolution_clock::now();
			auto screenshot = vmc->vm->screenshot();

			if (visit_select_expr(check->select_expr, screenshot)) {
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
		std::throw_with_nested(ActionException(check, vmc));
	}
}

std::string VisitorInterpreter::test_cksum(std::shared_ptr<AST::Test> test) const {
	VisitorCksum visitor(reg);
	return std::to_string(visitor.visit(test));
}

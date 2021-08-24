
#include <coro/CheckPoint.h>
#include <coro/Timeout.h>
#include "VisitorInterpreterActionFlashDrive.hpp"
#include "Exceptions.hpp"
#include <fmt/format.h>

void VisitorInterpreterActionFlashDrive::visit_action(std::shared_ptr<AST::Action> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Abort>(action)) {
		visit_abort({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Print>(action)) {
		visit_print({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Sleep>(action)) {
		visit_sleep({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Copy>(action)) {
		visit_copy({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Block<AST::Action>>(action)) {
		visit_action_block(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::ActionWithDelim>(action)) {
		visit_action(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Empty>(action)) {
		;
	} else if (auto p = std::dynamic_pointer_cast<AST::MacroCall<AST::Action>>(action)) {
		visit_macro_call({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::IfClause>(action)) {
		visit_if_clause(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::ForClause>(action)) {
		visit_for_clause(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::CycleControl>(action)) {
		throw CycleControlException(p->token);
	}  else {
		throw std::runtime_error("Should never happen");
	}

	coro::CheckPoint();
}

void VisitorInterpreterActionFlashDrive::visit_copy(const IR::Copy& copy) {
	try {
		fs::path from = copy.from();
		fs::path to = copy.to();

		IR::TimeInterval wait_for = copy.timeout();
		reporter.copy(current_controller, from.generic_string(), to.generic_string(), copy.ast_node->is_to_guest(), wait_for.str());

		coro::Timeout timeout(wait_for.value());

		for (auto vmc: current_test->get_all_machines()) {
			if (vmc->vm()->is_flash_plugged(fdc->fd())) {
				throw std::runtime_error(fmt::format("Flash drive {} is already plugged into vm {}. You should unplug it first", fdc->name(), vmc->name()));
			}
		}

		//TODO: timeouts
		if(copy.ast_node->is_to_guest()) {
			//Additional check since now we can't be sure the "from" actually exists
			if (!fs::exists(from)) {
				throw std::runtime_error(std::string(copy.ast_node->begin()) + ": Error: specified path doesn't exist: " + from.generic_string());
			}
			fdc->fd()->upload(from, to);
		} else {
			fdc->fd()->download(from, to);
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(copy.ast_node, current_controller));
	}
}

bool VisitorInterpreterActionFlashDrive::visit_check(const IR::Check& check) {
	throw std::runtime_error("Shouldn't get here");
}

void VisitorInterpreterActionFlashDrive::visit_abort(const IR::Abort& abort) {
	throw AbortException(abort.ast_node, current_controller, abort.message());
}


#include "RegisterUtils.hpp"
#include "TemplateParser.hpp"
#include "StackEntry.hpp"
#include "coro/Finally.h"

std::set<std::shared_ptr<Controller>> get_all_controllers(std::shared_ptr<AST::Test> test, std::shared_ptr<Register> reg) {
	std::set<std::shared_ptr<Controller>> result;

	auto vmcs = get_all_vmcs(test, reg);
	result.insert(vmcs.begin(), vmcs.end());

	auto fdcs = get_all_fdcs(test, reg);
	result.insert(fdcs.begin(), fdcs.end());

	return result;
}

std::set<std::shared_ptr<VmController>> get_all_vmcs(std::shared_ptr<AST::Test> test, std::shared_ptr<Register> reg) {
	std::set<std::shared_ptr<VmController>> result;
	for (auto parent: test->parents) {
		for (auto vmc: get_all_vmcs(parent, reg)) {
			result.insert(vmc);
		}
	}

	//Now to go through all the commands

	for (auto command: test->cmd_block->commands) {
		for (auto vm_token: command->vms) {
			auto vmc_request = reg->vmc_requests.find(vm_token.value());
			if (vmc_request == reg->vmc_requests.end()) {
				throw std::runtime_error("get_all_vmcs"); //should never happen
			}
			result.insert(vmc_request->second.get_vmc());
		}
	}

	return result;
}

std::set<std::shared_ptr<NetworkController>> get_all_netcs(std::shared_ptr<AST::Test> test, std::shared_ptr<Register> reg) {
	std::set<std::shared_ptr<NetworkController>> result;
	for (auto vmc: get_all_vmcs(test, reg)) {
		for (std::string network_name: vmc->vm->networks()) {
			auto netc = reg->netcs.find(network_name);
			if (netc == reg->netcs.end()) {
				throw std::runtime_error("get_all_netcs"); //should never happen
			}
			result.insert(netc->second);
		}
	}

	return result;
}

std::set<std::shared_ptr<FlashDriveController>> extract_fdcs_from_action(std::shared_ptr<AST::IAction> action, std::shared_ptr<Register> reg) {
	std::set<std::shared_ptr<FlashDriveController>> result;

	template_literals::Parser template_parser;

	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Plug>>(action)) {
		if (p->action->type.value() == "flash") {
			template_literals::Parser template_parser;
			auto name = template_parser.resolve(p->action->name->text(), reg);
			result.insert(reg->fdc_requests.find(name)->second.get_fdc());
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		for (auto action: p->action->actions) {
			auto tmp = extract_fdcs_from_action(action, reg);
			result.insert(tmp.begin(), tmp.end());
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroActionCall>>(action)) {
		auto macro_action_call = p->action;

		//Okay now, we need to push local args

		//push new ctx
		StackEntry new_ctx(true);

		for (size_t i = 0; i < macro_action_call->args.size(); ++i) {
			auto value = template_parser.resolve(macro_action_call->args[i]->text(), reg);
			new_ctx.define(macro_action_call->macro_action->args[i]->name(), value);
		}

		for (size_t i = macro_action_call->args.size(); i < macro_action_call->macro_action->args.size(); ++i) {
			auto value = template_parser.resolve(macro_action_call->macro_action->args[i]->default_value->text(), reg);
			new_ctx.define(macro_action_call->macro_action->args[i]->name(), value);
		}

		reg->local_vars.push_back(new_ctx);
		coro::Finally finally([&] {
			reg->local_vars.pop_back();
		});

		auto tmp = extract_fdcs_from_action(p->action->macro_action->action_block, reg);
		result.insert(tmp.begin(), tmp.end());
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		auto for_clause = p->action;

		StackEntry new_ctx(false);
		reg->local_vars.push_back(new_ctx);
		size_t ctx_position = reg->local_vars.size() - 1;
		coro::Finally finally([&] {
			reg->local_vars.pop_back();
		});

		for (auto i: for_clause->counter_list->values()) {
			reg->local_vars[ctx_position].define(for_clause->counter.value(), i);
			auto tmp = extract_fdcs_from_action(for_clause->cycle_body, reg);
			result.insert(tmp.begin(), tmp.end());
		}

		if (for_clause->else_token) {
			auto tmp = extract_fdcs_from_action(for_clause->else_action, reg);
			result.insert(tmp.begin(), tmp.end());
		}
	} 

	return result;
}

std::set<std::shared_ptr<FlashDriveController>> get_all_fdcs(std::shared_ptr<AST::Test> test, std::shared_ptr<Register> reg) {
	std::set<std::shared_ptr<FlashDriveController>> result;
	for (auto parent: test->parents) {
		auto fdcs = get_all_fdcs(parent, reg);
		result.insert(fdcs.begin(), fdcs.end());
	}

	//Now to go through all the commands

	for (auto command: test->cmd_block->commands) {
		auto fdcs = extract_fdcs_from_action(command->action, reg);
		result.insert(fdcs.begin(), fdcs.end());
	}
	return result;
}

std::list<std::shared_ptr<AST::Test>> get_test_path(std::shared_ptr<AST::Test> test, std::shared_ptr<Register> reg) {
	std::list<std::shared_ptr<AST::Test>> result;

	/*
		For every parent. Get his parents. Iterate though them. If we already have a parent - don't include it
		Then do the same check for the parents themselves
	*/

	for (auto parent: test->parents) {
		concat_unique(result, get_test_path(parent, reg));
	}

	concat_unique(result, test->parents);
	concat_unique(result, {test});

	return result;
}

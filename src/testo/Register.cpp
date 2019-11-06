
#include "Register.hpp"

Register::~Register() {
	for (auto fdc: fdcs) {
		if (fdc.second->fd->is_mounted()) {
			fdc.second->fd->umount();
		}
	}
}

std::set<std::shared_ptr<Controller>> Register::get_all_controllers(std::shared_ptr<AST::Test> test) const {
	std::set<std::shared_ptr<Controller>> result;

	auto vmcs = get_all_vmcs(test);
	result.insert(vmcs.begin(), vmcs.end());

	auto fdcs = get_all_fdcs(test);
	result.insert(fdcs.begin(), fdcs.end());

	return result;
}

std::set<std::shared_ptr<VmController>> Register::get_all_vmcs(std::shared_ptr<AST::Test> test) const {
	std::set<std::shared_ptr<VmController>> result;
	for (auto parent: test->parents) {
		for (auto vmc: get_all_vmcs(parent)) {
			result.insert(vmc);
		}
	}

	//Now to go through all the commands

	for (auto command: test->cmd_block->commands) {
		for (auto vm_token: command->vms) {
			auto vmc = vmcs.find(vm_token.value());
			if (vmc == vmcs.end()) {
				throw std::runtime_error("get_all_vmcs"); //should never happen
			}
			result.insert(vmc->second);
		}
	}

	return result;
}

std::set<std::shared_ptr<NetworkController>> Register::get_all_netcs(std::shared_ptr<AST::Test> test) const {
	std::set<std::shared_ptr<NetworkController>> result;
	for (auto vmc: get_all_vmcs(test)) {
		for (std::string network_name: vmc->vm->networks()) {
			auto netc = netcs.find(network_name);
			if (netc == netcs.end()) {
				throw std::runtime_error("get_all_netcs"); //should never happen
			}
			result.insert(netc->second);
		}
	}

	return result;
}

std::set<std::shared_ptr<FlashDriveController>> Register::extract_fdcs_from_action(std::shared_ptr<AST::IAction> action) const {
	std::set<std::shared_ptr<FlashDriveController>> result;

	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Plug>>(action)) {
		if (p->action->type.value() == "flash") {
			result.insert(fdcs.find(p->action->name_token.value())->second);
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		for (auto action: p->action->actions) {
			auto tmp = extract_fdcs_from_action(action);
			result.insert(tmp.begin(), tmp.end());
		}
	}

	return result;
}

std::set<std::shared_ptr<FlashDriveController>> Register::get_all_fdcs(std::shared_ptr<AST::Test> test) const {
	std::set<std::shared_ptr<FlashDriveController>> result;
	for (auto parent: test->parents) {
		auto fdcs = get_all_fdcs(parent);
		result.insert(fdcs.begin(), fdcs.end());
	}

	//Now to go through all the commands

	for (auto command: test->cmd_block->commands) {
		auto fdcs = extract_fdcs_from_action(command->action);
		result.insert(fdcs.begin(), fdcs.end());
	}
	return result;
}

std::list<std::shared_ptr<AST::Test>> Register::get_test_path(std::shared_ptr<AST::Test> test) const {
	std::list<std::shared_ptr<AST::Test>> result;

	/*
		For every parent. Get his parents. Iterate though them. If we already have a parent - don't include it
		Then do the same check for the parents themselves
	*/

	for (auto parent: test->parents) {
		concat_unique(result, get_test_path(parent));
	}

	concat_unique(result, test->parents);
	concat_unique(result, {test});

	return result;
}

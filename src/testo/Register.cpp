
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

	for (auto vmc: get_all_vmcs(test)) {
		result.insert(vmc);
	}

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
				throw std::runtime_error(""); //should never happen
			}
			result.insert(vmc->second);
		}
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

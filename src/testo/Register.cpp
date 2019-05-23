
#include "Register.hpp"

Register::~Register() {
	for (auto fd: fds) {
		if (fd.second->is_mounted()) {
			fd.second->umount();
		}
	}
}

std::set<std::shared_ptr<VmController>> Register::get_all_vms(std::shared_ptr<AST::Test> test) const {
	std::set<std::shared_ptr<VmController>> result;
	for (auto parent: test->parents) {
		for (auto vm: get_all_vms(parent)) {
			result.insert(vm);
		}
	}

	//Now to go through all the commands

	for (auto command: test->cmd_block->commands) {
		for (auto vm_token: command->vms) {
			auto vm = vms.find(vm_token.value());
			if (vm == vms.end()) {
				throw std::runtime_error(""); //should never happen
			}
			result.insert(vm->second);
		}
	}

	return result;
}

std::vector<std::shared_ptr<AST::Test>> Register::get_test_path(std::shared_ptr<AST::Test> test) const {
	std::vector<std::shared_ptr<AST::Test>> result;

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

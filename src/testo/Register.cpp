
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
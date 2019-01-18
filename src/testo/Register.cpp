
#include "Register.hpp"

Register::~Register() {
	for (auto fd: fds) {
		if (fd.second->is_mounted()) {
			fd.second->umount();
		}
	}

	for (auto vm: vms) {
		while (!vm.second->plugged_fds.empty()) {
			vm.second->unplug_flash_drive(*vm.second->plugged_fds.begin());
		}
	}
}
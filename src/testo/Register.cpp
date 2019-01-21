
#include "Register.hpp"

Register::~Register() {
	for (auto fd: fds) {
		if (fd.second->is_mounted()) {
			fd.second->umount();
		}
	}

	for (auto vm: vms) {
		vm.second->unplug_all_flash_drives();
	}
}
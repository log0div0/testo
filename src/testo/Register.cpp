
#include "Register.hpp"

std::shared_ptr<FlashDriveController> FlashDriveControllerRequest::get_fdc() {
	if (!fdc) {
		fdc = env->create_flash_drive_controller(config);

		if (fdc->fd->has_folder()) {
			fdc->fd->validate_folder();
		}
	}

	return fdc;
}

std::shared_ptr<VmController> VmControllerRequest::get_vmc() {
	if (!vmc) {
		vmc = env->create_vm_controller(config);

		//additional check that all the networks are defined earlier
		for (auto network: vmc->vm->networks()) {
			if (reg->netcs.find(network) == reg->netcs.end()) {
				throw std::runtime_error("Error constructing VM " + vmc->name() + ": network " + network + " is not defined");
			}
		}
	}

	return vmc;
}

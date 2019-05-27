
#include "VboxEnvironment.hpp"
#include "VboxVmController.hpp"
#include "VboxFlashDriveController.hpp"

#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>

fs::path VboxEnvironment::flash_drives_img_dir;
fs::path VboxEnvironment::flash_drives_mount_dir;

VboxEnvironment::VboxEnvironment() {
	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box = virtual_box_client.virtual_box();
	auto path = fs::path(virtual_box.compose_machine_filename("testo", "/", {}, {}));
	auto flash_drive_dir = path.parent_path().parent_path().parent_path() / "VirtualBox Flash Drives";
	flash_drives_img_dir = flash_drive_dir / "images";
	flash_drives_mount_dir = flash_drive_dir / "mount_point";
}

VboxEnvironment::~VboxEnvironment() {
	try {
		cleanup();
	} catch (...) {}
}


void VboxEnvironment::setup() {
	cleanup();

	if (std::system("lsmod | grep nbd > /dev/null")) {
		throw std::runtime_error("Please load nbd module (max parts=1");
	}

	if (!fs::exists(flash_drives_img_dir)) {
		if (!fs::create_directories(flash_drives_img_dir)) {
			throw std::runtime_error(std::string("Can't create directory: ") + flash_drives_img_dir.generic_string());
		}
	}

	if (!fs::exists(flash_drives_mount_dir)) {
		if (!fs::create_directories(flash_drives_mount_dir)) {
			throw std::runtime_error(std::string("Can't create directory: ") + flash_drives_mount_dir.generic_string());
		}
	}
}

void VboxEnvironment::cleanup() {
	std::string fdisk = std::string("fdisk -l | grep nbd0");
	if (std::system(fdisk.c_str()) == 0) {
		exec_and_throw_if_failed(std::string("qemu-nbd --disconnect /dev/nbd0"));
	}

	if (!fs::exists(flash_drives_img_dir)) {
		return;
	}

	vbox::VirtualBoxClient virtual_box_client;
	auto virtual_box = virtual_box_client.virtual_box();
	auto hdds = virtual_box.hard_disks();

	for (auto& p: fs::directory_iterator(flash_drives_img_dir)) {
		for (auto& hdd: hdds) {
			if (fs::path(p).generic_string() == hdd.location()) {
				hdd.delete_storage().wait_and_throw_if_failed();
				break;
			}
		}
	}
}

std::shared_ptr<VmController> VboxEnvironment::create_vm_controller(const nlohmann::json& config) {
	return std::shared_ptr<VmController>(new VboxVmController(config));
}

std::shared_ptr<FlashDriveController> VboxEnvironment::create_flash_drive_controller(const nlohmann::json& config) {
	return std::shared_ptr<FlashDriveController>(new VboxFlashDriveController(config));
}

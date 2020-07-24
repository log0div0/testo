
#include "VboxEnvironment.hpp"
#include "VboxVM.hpp"
#include "VboxFlashDrive.hpp"
#include "VboxNetwork.hpp"

#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>

VboxEnvironment::VboxEnvironment() {
#ifdef WIN32
	_putenv_s("VBOX", "1");
#else
	setenv("VBOX", "1", false);
#endif
	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box = virtual_box_client.virtual_box();
	auto path = fs::path(virtual_box.compose_machine_filename("testo", "/", {}, {}));
	auto flash_drive_dir = path.parent_path().parent_path().parent_path() / "VirtualBox Flash Drives";
	_flash_drives_img_dir = flash_drive_dir / "images";
	_flash_drives_mount_dir = flash_drive_dir / "mount_point";
}

VboxEnvironment::~VboxEnvironment() {
	cleanup();
}


void VboxEnvironment::setup() {
	cleanup();

	if (!fs::exists(_flash_drives_img_dir)) {
		if (!fs::create_directories(_flash_drives_img_dir)) {
			throw std::runtime_error(std::string("Can't create directory: ") + _flash_drives_img_dir.generic_string());
		}
	}

	if (!fs::exists(_flash_drives_mount_dir)) {
		if (!fs::create_directories(_flash_drives_mount_dir)) {
			throw std::runtime_error(std::string("Can't create directory: ") + _flash_drives_mount_dir.generic_string());
		}
	}
}

void VboxEnvironment::cleanup() {
}

std::shared_ptr<VM> VboxEnvironment::create_vm(const nlohmann::json& config) {
	return std::shared_ptr<VM>(new VboxVM(config));
}

std::shared_ptr<FlashDrive> VboxEnvironment::create_flash_drive(const nlohmann::json& config) {
	return std::shared_ptr<FlashDrive>(new VboxFlashDrive(config));
}

std::shared_ptr<Network> VboxEnvironment::create_network(const nlohmann::json& config) {
	return std::shared_ptr<Network>(new VboxNetwork(config));
}

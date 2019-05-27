
#include "VboxEnvironment.hpp"

#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>

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

	exec_and_throw_if_failed("mkdir -p " + flash_drives_img_dir().generic_string());
	exec_and_throw_if_failed("mkdir -p " + flash_drives_mount_dir().generic_string());
}

void VboxEnvironment::cleanup() {
	std::string fdisk = std::string("fdisk -l | grep nbd0");
	if (std::system(fdisk.c_str()) == 0) {
		exec_and_throw_if_failed(std::string("qemu-nbd --disconnect /dev/nbd0"));
	}

	if (!fs::exists(flash_drives_img_dir())) {
		return;
	}

	vbox::VirtualBoxClient virtual_box_client;
	auto virtual_box = virtual_box_client.virtual_box();
	auto hdds = virtual_box.hard_disks();

	auto img_dir = flash_drives_img_dir();
	for (auto& p: fs::directory_iterator(img_dir)) {
		for (auto& hdd: hdds) {
			if (fs::path(p).generic_string() == hdd.location()) {
				hdd.delete_storage().wait_and_throw_if_failed();
				break;
			}
		}
	}

	//now we need to close all the open

	exec_and_throw_if_failed("rm -rf " + flash_drives_img_dir().generic_string());
}

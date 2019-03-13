
#include "Environment.hpp"
#include <fmt/format.h>
#include <iostream>
#include "Utils.hpp"
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
	exec_and_throw_if_failed("mkdir -p " + scripts_tmp_dir().generic_string());
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

QemuEnvironment::~QemuEnvironment() {
	try {
		cleanup();
	} catch(...) {}
}

void QemuEnvironment::prepare_storage_pool(const std::string& pool_name) {
	auto pool_dir = testo_dir() / pool_name;
	if (!fs::exists(pool_dir)) {
		if (!fs::create_directory(pool_dir)) {
			throw std::runtime_error(std::string("Can't create directory: ") + pool_dir.generic_string());
		}
	}

	if (!fs::exists(scripts_tmp_dir())) {
		if (!fs::create_directory(scripts_tmp_dir())) {
			throw std::runtime_error(std::string("Can't create directory: ") + pool_dir.generic_string());
		}
	}


	auto storage_pools = qemu_connect.storage_pools({VIR_CONNECT_LIST_STORAGE_POOLS_PERSISTENT});

	bool found = false;
	for (auto& pool: storage_pools) {
		if (pool.name() == pool_name) {
			if (!pool.is_active()) {
				std::cout << "INFO: " << pool_name <<  "is inactive, starting...\n";
			}
			found = true;
			break;
		}
	}

	if (!found) {
		std::cout << "INFO: " << pool_name <<  "is not found, creating...\n";
		pugi::xml_document xml_config;
		xml_config.load_string(fmt::format(R"(
			<pool type='dir'>
				<name>{}</name>
				<source>
				</source>
				<target>
					<path>{}</path>
					<permissions>
						<mode>0775</mode>
						<owner>1000</owner>
						<group>1000</group>
					</permissions>
				</target>
			</pool>
		)", pool_name, pool_dir.generic_string()).c_str());
		auto pool = qemu_connect.storage_pool_define_xml(xml_config);
		pool.start({VIR_STORAGE_POOL_CREATE_NORMAL});
	}
}

void QemuEnvironment::setup() {
	qemu_connect = vir::connect_open("qemu:///system");
	prepare_storage_pool("testo-storage-pool");
	prepare_storage_pool("testo-flash-drives-pool");

	//TODO: fuck that mounting
	exec_and_throw_if_failed("mkdir -p " + flash_drives_mount_dir().generic_string());
}

void QemuEnvironment::cleanup() {

}

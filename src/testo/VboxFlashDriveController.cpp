
#include "VboxFlashDriveController.hpp"
#include "Utils.hpp"
#include <functional>
#include <thread>
#include <chrono>

VboxFlashDriveController::VboxFlashDriveController(const nlohmann::json& config):
config(config), api(API::instance())
{
	if (!config.count("name")) {
		throw std::runtime_error("Constructing VboxFlashDriveController error: field NAME is not specified");
	}

	if (!config.count("size")) {
		throw std::runtime_error("Constructing VboxFlashDriveController error: field SIZE is not specified");
	}

	//TODO: check for fs types
	if (!config.count("fs")) {
		throw std::runtime_error("Constructing VboxFlashDriveController error: field FS is not specified");
	}

	virtual_box = virtual_box_client.virtual_box();
}

int VboxFlashDriveController::create() {
	try {
		std::string fdisk = std::string("fdisk -l | grep nbd0");
		if (std::system(fdisk.c_str()) == 0) {
			throw std::runtime_error("Can't create flash drive: target host slot is busy");
		}

		handle = virtual_box.create_medium("vmdk", img_path().generic_string(), AccessMode_ReadWrite, DeviceType_HardDisk);
		size_t disk_size = config.at("size").get<uint32_t>();
		disk_size = disk_size * 1024 * 1024;
		handle.create_base_storage(disk_size, MediumVariant_Fixed).wait_and_throw_if_failed();

		exec_and_throw_if_failed(std::string("qemu-nbd --connect=") +
			"/dev/nbd0 -f vmdk " +
			img_path().generic_string());

		std::string size = std::to_string(config.at("size").get<uint32_t>()) + "M";
		exec_and_throw_if_failed(std::string("parted --script -a optimal /dev/nbd0 mklabel msdos mkpart primary 0% ") +
			size);

		exec_and_throw_if_failed(std::string("mkfs.") +
			config.at("fs").get<std::string>() +
			" /dev/nbd0");

		exec_and_throw_if_failed(std::string("qemu-nbd -d /dev/nbd0"));
		return 0;
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
		return 1;
	}
}

bool VboxFlashDriveController::is_mounted() const {
	std::string query = std::string("mountpoint -q " + flash_drives_mount_dir().generic_string());
	return (std::system(query.c_str()) == 0);
}

int VboxFlashDriveController::mount() const {
	try {
		std::string fdisk = std::string("fdisk -l | grep nbd0");
		if (std::system(fdisk.c_str()) == 0) {
			throw std::runtime_error("Can't mount flash drive: target host slot is busy");
		}

		exec_and_throw_if_failed(std::string("qemu-nbd --connect=") +
			"/dev/nbd0 -f vmdk " +
			img_path().generic_string());

		exec_and_throw_if_failed(std::string("mount /dev/nbd0"));
		return 0;
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
		return 1;
	}
}

int VboxFlashDriveController::umount() const {
	try {
		exec_and_throw_if_failed(std::string("umount /dev/nbd0"));
		exec_and_throw_if_failed(std::string("qemu-nbd -d /dev/nbd0"));
		return 0;
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
		return 1;
	}
}

int VboxFlashDriveController::load_folder() const {
	try {
		fs::path target_folder(config.at("folder").get<std::string>());

		auto abs_target_folder = std::experimental::filesystem::absolute(target_folder);

		if (!fs::exists(abs_target_folder)) {
			throw std::runtime_error("Target folder doesn't exist");
		}
		if (mount()) {
			throw std::runtime_error("performing mount while loading folder");
		}

		exec_and_throw_if_failed(std::string("cp -r ") +
			abs_target_folder.generic_string() +
			" " + flash_drives_mount_dir().generic_string());
		std::this_thread::sleep_for(std::chrono::seconds(2));
		if (umount()) {
			throw std::runtime_error("performing mount while loading folder");
		}
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Load folder error: " << error << std::endl;
		return 1;
	}


}

#include "VboxFlashDriveController.hpp"
#include "../../Utils.hpp"
#include <functional>
#include <thread>
#include <chrono>

VboxFlashDriveController::VboxFlashDriveController(const nlohmann::json& config_): FlashDriveController(config_)
{
	virtual_box = virtual_box_client.virtual_box();
}

void VboxFlashDriveController::create() {
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
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
	}
}

bool VboxFlashDriveController::is_mounted() const {
	std::string query = std::string("mountpoint -q " + flash_drives_mount_dir().generic_string());
	return (std::system(query.c_str()) == 0);
}

void VboxFlashDriveController::mount() const {
	try {
		std::string fdisk = std::string("fdisk -l | grep nbd0");
		if (std::system(fdisk.c_str()) == 0) {
			throw std::runtime_error("Can't mount flash drive: target host slot is busy");
		}

		exec_and_throw_if_failed(std::string("qemu-nbd --connect=") +
			"/dev/nbd0 -f vmdk " +
			img_path().generic_string());

		exec_and_throw_if_failed(std::string("mount /dev/nbd0"));
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
	}
}

void VboxFlashDriveController::umount() const {
	try {
		exec_and_throw_if_failed(std::string("umount /dev/nbd0"));
		exec_and_throw_if_failed(std::string("qemu-nbd -d /dev/nbd0"));
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
	}
}

void VboxFlashDriveController::load_folder() const {
	try {
		fs::path target_folder(config.at("folder").get<std::string>());

		auto abs_target_folder = std::experimental::filesystem::absolute(target_folder);

		if (!fs::exists(abs_target_folder)) {
			throw std::runtime_error("Target folder doesn't exist");
		}
		mount();

		exec_and_throw_if_failed(std::string("cp -r ") +
			abs_target_folder.generic_string() +
			" " + flash_drives_mount_dir().generic_string());
		std::this_thread::sleep_for(std::chrono::seconds(2));
		umount();
	} catch (const std::exception& error) {
		std::cout << "Load folder error: " << error << std::endl;
	}
}

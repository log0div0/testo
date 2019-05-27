
#include "VboxFlashDriveController.hpp"
#include "VboxEnvironment.hpp"
#include <functional>
#include <thread>
#include <chrono>

#ifdef __linux__
const std::string disk_format = "vmdk";
#else
const std::string disk_format = "vhd";
#endif

VboxFlashDriveController::VboxFlashDriveController(const nlohmann::json& config_): FlashDriveController(config_)
{
	virtual_box = virtual_box_client.virtual_box();
}

void VboxFlashDriveController::create() {
	try {
#ifdef __linux__
		if (std::system("lsmod | grep nbd > /dev/null")) {
			throw std::runtime_error("Please load nbd module (max parts=1");
		}
#endif
		remove_if_exists();

		auto handle = virtual_box.create_medium(disk_format, img_path().generic_string(), AccessMode_ReadWrite, DeviceType_HardDisk);
		size_t disk_size = config.at("size").get<uint32_t>();
		disk_size = disk_size * 1024 * 1024;
		handle.create_base_storage(disk_size, MediumVariant_Fixed).wait_and_throw_if_failed();

#ifdef __linux__
		exec_and_throw_if_failed(std::string("qemu-nbd --connect=") +
			"/dev/nbd0 -f " + disk_format +
			" \"" + img_path().generic_string() + "\"");

		std::string size = std::to_string(config.at("size").get<uint32_t>()) + "M";
		exec_and_throw_if_failed(std::string("parted --script -a optimal /dev/nbd0 mklabel msdos mkpart primary 0% ") +
			size);

		exec_and_throw_if_failed(std::string("mkfs.") +
			config.at("fs").get<std::string>() +
			" /dev/nbd0");

		exec_and_throw_if_failed(std::string("qemu-nbd -d /dev/nbd0"));
#endif
		write_cksum(calc_cksum());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool VboxFlashDriveController::is_mounted() const {
#ifdef __linux__
	std::string query = std::string("mountpoint -q " + VboxEnvironment::flash_drives_mount_dir.generic_string());
	return (std::system(query.c_str()) == 0);
#endif
	return false;
}

void VboxFlashDriveController::mount() const {
	try {
#ifdef __linux__
		std::string fdisk = std::string("fdisk -l | grep nbd0");
		if (std::system(fdisk.c_str()) == 0) {
			throw std::runtime_error("Can't mount flash drive: target host slot is busy");
		}

		exec_and_throw_if_failed(std::string("qemu-nbd --connect=") +
			"/dev/nbd0 -f " + disk_format +
			" \"" + img_path().generic_string() + "\"");

		exec_and_throw_if_failed(std::string("mount /dev/nbd0"));
#endif
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxFlashDriveController::umount() const {
	try {
#ifdef __linux__
		exec_and_throw_if_failed(std::string("umount /dev/nbd0"));
		exec_and_throw_if_failed(std::string("qemu-nbd -d /dev/nbd0"));
#endif
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
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

#ifdef __linux__
		exec_and_throw_if_failed(std::string("cp -r ") +
			abs_target_folder.generic_string() +
			" " + VboxEnvironment::flash_drives_mount_dir.generic_string());
		std::this_thread::sleep_for(std::chrono::seconds(2));
#endif

		umount();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

fs::path VboxFlashDriveController::img_path() const {
	return VboxEnvironment::flash_drives_img_dir / (name() + "." + disk_format);
}

void VboxFlashDriveController::remove_if_exists() {
	try {
		for (auto& hdd: virtual_box.hard_disks()) {
			if (img_path().generic_string() == hdd.location()) {
				hdd.delete_storage().wait_and_throw_if_failed();
				break;
			}
		}

		delete_cksum();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

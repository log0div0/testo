
#include "VboxFlashDrive.hpp"
#include "VboxEnvironment.hpp"
#include <functional>
#include <thread>
#include <chrono>

#ifdef __linux__
const std::string disk_format = "vmdk";
#elif WIN32
#include <msft/Connect.hpp>
#include <virtdisk/VirtualDisk.hpp>
const std::string disk_format = "vhd";
#elif __APPLE__
const std::string disk_format = "????";
#endif

VboxFlashDrive::VboxFlashDrive(const nlohmann::json& config_): FlashDrive(config_)
{
	virtual_box = virtual_box_client.virtual_box();
}

bool VboxFlashDrive::is_defined() {
	throw std::runtime_error("Implement me");
}

void VboxFlashDrive::create() {
	try {
#ifdef __linux__
		if (std::system("lsmod | grep nbd > /dev/null")) {
			throw std::runtime_error("Please load nbd module (max parts=1)");
		}
#endif
		remove_if_exists();

		auto handle = virtual_box.create_medium(disk_format, img_path().generic_string(), AccessMode_ReadWrite, DeviceType_HardDisk);
		size_t disk_size = config.at("size").get<uint32_t>();
		disk_size = disk_size * 1024 * 1024;
		handle.create_base_storage(disk_size, MediumVariant_Fixed).wait_and_throw_if_failed();

#ifdef __linux__
		exec_and_throw_if_failed("qemu-nbd --connect=/dev/nbd0 -f " + disk_format + " \"" + img_path().generic_string() + "\"");
		exec_and_throw_if_failed("parted --script -a optimal /dev/nbd0 mklabel msdos mkpart primary 0% 100%");
		exec_and_throw_if_failed("mkfs." + config.at("fs").get<std::string>() + " /dev/nbd0p1");
		exec_and_throw_if_failed("qemu-nbd -d /dev/nbd0");
#elif WIN32
		VirtualDisk virtualDisk(img_path().generic_string());
		virtualDisk.attach();
		msft::Connect connect;
		auto disk = connect.virtualDisk(img_path().generic_string());
		disk.initialize();
		disk.createPartition();
		auto partition = disk.partitions().at(0);
		auto volume = partition.volume();
		volume.format("NTFS", name());
		virtualDisk.detach();
#endif
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxFlashDrive::undefine() {
	throw std::runtime_error("Implement me");
}


bool VboxFlashDrive::is_mounted() const {
#ifdef __linux__
	std::string query = "mountpoint -q \"" + env->flash_drives_mount_dir().generic_string() + "\"";
	return (std::system(query.c_str()) == 0);
#elif WIN32
	return VirtualDisk(img_path().generic_string()).isLoaded();
#elif __APPLE__
	throw std::runtime_error(__PRETTY_FUNCTION__);
#endif
}

void VboxFlashDrive::mount() {
	try {
#ifdef __linux__
		std::string fdisk = "fdisk -l | grep nbd0";
		if (std::system(fdisk.c_str()) == 0) {
			throw std::runtime_error("Can't mount flash drive: target host slot is busy");
		}

		exec_and_throw_if_failed("qemu-nbd --connect=/dev/nbd0 -f " + disk_format +	" \"" + img_path().generic_string() + "\"");
		exec_and_throw_if_failed("mount /dev/nbd0");
#elif WIN32
		VirtualDisk virtualDisk(img_path().generic_string());
		virtualDisk.attach();
		msft::Connect connect;
		auto disk = connect.virtualDisk(img_path().generic_string());
		auto partition = disk.partitions().at(0);
		partition.addAccessPath(env->flash_drives_mount_dir().generic_string());
#endif
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxFlashDrive::umount() {
	try {
#ifdef __linux__
		exec_and_throw_if_failed("umount /dev/nbd0");
		exec_and_throw_if_failed("qemu-nbd -d /dev/nbd0");
#elif WIN32
		msft::Connect connect;
		auto disk = connect.virtualDisk(img_path().generic_string());
		auto partition = disk.partitions().at(0);
		partition.removeAccessPath(env->flash_drives_mount_dir().generic_string());
		VirtualDisk virtualDisk(img_path().generic_string());
		virtualDisk.detach();
#endif
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool VboxFlashDrive::has_snapshot(const std::string& snapshot) {
	throw std::runtime_error("Implement me");
}

void VboxFlashDrive::make_snapshot(const std::string& snapshot) {
	throw std::runtime_error("Implement me");
}

void VboxFlashDrive::delete_snapshot(const std::string& snapshot) {
	throw std::runtime_error("Implement me");
}

void VboxFlashDrive::rollback(const std::string& snapshot) {
	throw std::runtime_error("Implement me");
}

fs::path VboxFlashDrive::img_path() const {
	return env->flash_drives_img_dir() / (id() + "." + disk_format);
}

void VboxFlashDrive::remove_if_exists() {
	try {
		for (auto& hdd: virtual_box.hard_disks()) {
			if (img_path().generic_string() == hdd.location()) {
				hdd.delete_storage().wait_and_throw_if_failed();
				break;
			}
		}

		if (fs::exists(img_path())) {
			fs::remove(img_path());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}


#include "VboxFlashDrive.hpp"
#include "VboxEnvironment.hpp"
#include <functional>
#include <thread>
#include <chrono>
#include "process/Process.hpp"

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
		if (Process::is_failed("lsmod | grep nbd")) {
			throw std::runtime_error("Please load nbd module (max parts=1)");
		}
#endif
		remove_if_exists();

		auto handle = virtual_box.create_medium(disk_format, img_path().generic_string(), AccessMode_ReadWrite, DeviceType_HardDisk);
		size_t disk_size = config.at("size").get<uint32_t>();
		disk_size = disk_size * 1024 * 1024;
		handle.create_base_storage(disk_size, MediumVariant_Fixed).wait_and_throw_if_failed();

#ifdef __linux__
		Process::exec("qemu-nbd --connect=/dev/nbd0 -f " + disk_format + " \"" + img_path().generic_string() + "\"");
		Process::exec("parted --script -a optimal /dev/nbd0 mklabel msdos mkpart primary 0% 100%");
		Process::exec("mkfs." + config.at("fs").get<std::string>() + " /dev/nbd0p1");
		Process::exec("qemu-nbd -d /dev/nbd0");
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

void VboxFlashDrive::load_folder(const fs::path& folder) {
	throw std::runtime_error("Implement me");
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

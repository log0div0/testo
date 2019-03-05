
#include "pugixml/pugixml.hpp"
#include <fmt/format.h>
#include "QemuFlashDriveController.hpp"
#include "Utils.hpp"
#include <thread>

QemuFlashDriveController::QemuFlashDriveController(const nlohmann::json& config):
config(config), qemu_connect(vir::connect_open("qemu:///system"))
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
}

int QemuFlashDriveController::create() {
	try {
		if (std::system("lsmod | grep nbd > /dev/null")) {
			throw std::runtime_error("Please load nbd module (max parts=1");
		}

		remove_if_exists();

		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");
		pugi::xml_document xml_config;
		//TODO: Mode should be default!
		xml_config.load_string(fmt::format(R"(
			<volume type='file'>
				<name>{}.img</name>
				<source>
				</source>
				<capacity unit='M'>{}</capacity>
				<target>
					<path>{}</path>
					<format type='qcow2'/>
					<permissions>
					</permissions>
					<timestamps>
					</timestamps>
					<compat>1.1</compat>
					<features>
						<lazy_refcounts/>
					</features>
				</target>
			</volume>
		)", name(), config.at("size").get<uint32_t>(), full_img_path().generic_string()).c_str());

		auto volume = pool.volume_create_xml(xml_config, {VIR_STORAGE_VOL_CREATE_PREALLOC_METADATA});

		exec_and_throw_if_failed(std::string("qemu-nbd --connect=") +
			"/dev/nbd0 -f qcow2 " +
			full_img_path().generic_string());

		std::string size = std::to_string(config.at("size").get<uint32_t>()) + "M";
		exec_and_throw_if_failed(std::string("parted --script -a optimal /dev/nbd0 mklabel msdos mkpart primary 0% ") +
			size);

		exec_and_throw_if_failed(std::string("mkfs.") +
			config.at("fs").get<std::string>() +
			" /dev/nbd0");

		exec_and_throw_if_failed(std::string("qemu-nbd -d /dev/nbd0"));
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Creating flash drive " << name() << " error: " << error << std::endl;
		return 1;
	}
}

bool QemuFlashDriveController::is_mounted() const {
	std::string query = std::string("mountpoint -q " + flash_drives_mount_dir().generic_string());
	return (std::system(query.c_str()) == 0);
}

int QemuFlashDriveController::mount() const {
	try {
		std::string fdisk = std::string("fdisk -l | grep nbd0");
		if (std::system(fdisk.c_str()) == 0) {
			throw std::runtime_error("Can't mount flash drive: target host slot is busy");
		}

		exec_and_throw_if_failed(std::string("qemu-nbd --connect=") +
			"/dev/nbd0 -f qcow2 " +
			full_img_path().generic_string());

		exec_and_throw_if_failed(std::string("mount /dev/nbd0 ") + flash_drives_mount_dir().generic_string());
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Flash drive " << name() << " mount to host error: " << error << std::endl;
		return 1;
	}
}

int QemuFlashDriveController::umount() const {
		try {
		exec_and_throw_if_failed(std::string("umount /dev/nbd0"));
		exec_and_throw_if_failed(std::string("qemu-nbd -d /dev/nbd0"));
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Flash drive " << name() << " umount from host error: " << error << std::endl;
		return -1;
	}
}

int QemuFlashDriveController::load_folder() const {
	try {
		fs::path target_folder(config.at("folder").get<std::string>());

		if (target_folder.is_relative()) {
			target_folder = fs::canonical(target_folder);
		}

		if (!fs::exists(target_folder)) {
			throw std::runtime_error("Target folder doesn't exist");
		}
		if (mount()) {
			throw std::runtime_error("performing mount while loading folder");
		}

		exec_and_throw_if_failed(std::string("cp -r ") +
			target_folder.generic_string() +
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

void QemuFlashDriveController::remove_if_exists() {
	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");
	for (auto& vol: pool.volumes()) {
		if (vol.name() == (name() + ".img")) {
			vol.erase({VIR_STORAGE_VOL_DELETE_NORMAL});
		}
	}
}

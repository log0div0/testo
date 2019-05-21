
#include "pugixml/pugixml.hpp"
#include <fmt/format.h>
#include "QemuFlashDriveController.hpp"
#include "../../Utils.hpp"
#include <thread>
#include <fstream>

QemuFlashDriveController::QemuFlashDriveController(const nlohmann::json& config_): FlashDriveController(config_),
	qemu_connect(vir::connect_open("qemu:///system"))
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

	if (config.count("folder")) {
		fs::path folder(config.at("folder").get<std::string>());
		if (!fs::exists(folder)) {
			throw std::runtime_error(fmt::format("specified folder {} for flash drive {} does not exist",
				folder.generic_string(), name()));
		}

		if (!fs::is_directory(folder)) {
			throw std::runtime_error(fmt::format("specified folder {} for flash drive {} is not a folder",
				folder.generic_string(), name()));
		}
	}
}

void QemuFlashDriveController::create() {
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
		)", name(), config.at("size").get<uint32_t>(), img_path().generic_string()).c_str());

		auto volume = pool.volume_create_xml(xml_config, {VIR_STORAGE_VOL_CREATE_PREALLOC_METADATA});

		exec_and_throw_if_failed(std::string("qemu-nbd --connect=") +
			"/dev/nbd0 -f qcow2 " +
			img_path().generic_string());

		std::string size = std::to_string(config.at("size").get<uint32_t>()) + "M";
		exec_and_throw_if_failed(std::string("parted --script -a optimal /dev/nbd0 mklabel msdos mkpart primary 0% ") +
			size);

		exec_and_throw_if_failed(std::string("mkfs.") +
			config.at("fs").get<std::string>() +
			" /dev/nbd0");

		exec_and_throw_if_failed(std::string("qemu-nbd -d /dev/nbd0"));

		std::string cksum_input = name() + std::to_string(config.at("size").get<uint32_t>()) + config.at("fs").get<std::string>();
		if (has_folder()) {
			cksum_input += directory_signature(config.at("folder").get<std::string>());
		}

		std::hash<std::string> h;
		std::string cksum = std::to_string(h(cksum_input));

		fs::path cksum_path = img_dir() / (name() + ".cksum");
		std::ofstream output_stream(cksum_path, std::ofstream::out);
		if (!output_stream) {
			throw std::runtime_error(std::string("Can't create file for writing cksum: ") + cksum_path.generic_string());
		}
		output_stream << cksum;
		output_stream.close();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Creating flash drive"));
	}
}

bool QemuFlashDriveController::is_mounted() const {
	std::string query = std::string("mountpoint -q " + flash_drives_mount_dir().generic_string());
	return (std::system(query.c_str()) == 0);
}

void QemuFlashDriveController::mount() const {
	try {
		std::string fdisk = std::string("fdisk -l | grep nbd0");
		if (std::system(fdisk.c_str()) == 0) {
			throw std::runtime_error("Can't mount flash drive: target host slot is busy");
		}

		exec_and_throw_if_failed(std::string("qemu-nbd --connect=") +
			"/dev/nbd0 -f qcow2 " +
			img_path().generic_string());

		exec_and_throw_if_failed(std::string("mount /dev/nbd0 ") + flash_drives_mount_dir().generic_string());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive mount to host"));
	}
}

void QemuFlashDriveController::umount() const {
	try {
		exec_and_throw_if_failed(std::string("umount /dev/nbd0"));
		exec_and_throw_if_failed(std::string("qemu-nbd -d /dev/nbd0"));
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive umount from host"));
	}
}

void QemuFlashDriveController::load_folder() const {
	try {
		fs::path target_folder(config.at("folder").get<std::string>());

		if (target_folder.is_relative()) {
			target_folder = fs::canonical(target_folder);
		}

		if (!fs::exists(target_folder)) {
			throw std::runtime_error("Target folder doesn't exist");
		}
		mount();

		exec_and_throw_if_failed(std::string("cp -r ") +
			target_folder.generic_string() +
			" " + flash_drives_mount_dir().generic_string());
		std::this_thread::sleep_for(std::chrono::seconds(2));
		umount();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Load folder"));
	}
}

std::string QemuFlashDriveController::cksum() const {
	fs::path cksum_path = img_dir() / (name() + ".cksum");
	if (!fs::exists(cksum_path)) {
		return "";
	};

	if (!fs::is_regular_file(cksum_path)) {
		return "";
	};

	std::ifstream input_stream(cksum_path);

	if (!input_stream) {
		return "";
	}

	std::string result = std::string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
	return result;
}

void QemuFlashDriveController::remove_if_exists() {
	try {
		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");
		for (auto& vol: pool.volumes()) {
			if (vol.name() == (name() + ".img")) {
				vol.erase({VIR_STORAGE_VOL_DELETE_NORMAL});
			}
		}
		fs::path cksum_path = img_dir() / (name() + ".cksum");
		fs::remove(cksum_path);

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Remove flash if exist"));
	}

}

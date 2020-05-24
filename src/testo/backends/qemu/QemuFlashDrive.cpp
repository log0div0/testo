
#include "pugixml/pugixml.hpp"
#include <fmt/format.h>
#include "QemuFlashDrive.hpp"
#include "QemuEnvironment.hpp"
#include "coro/Timer.h"
#include "coro/Timeout.h"
#include "process/Process.hpp"
#include <thread>
#include <fstream>
#include <regex>
#include <linux/nbd.h>

using namespace std::chrono_literals;

QemuNbd::QemuNbd(const fs::path& img_path) {
	try {
		coro::Timeout timeout{30s};
		coro::Timer timer;

		while (Process::is_succeeded("fdisk -l | grep nbd0")) {
			timer.waitFor(2s);
		}

		Process::exec("qemu-nbd --connect=/dev/nbd0 -f qcow2 \"" + img_path.generic_string() + "\"");
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Trying plug flash drive to any free nbd slot"));
	}
}

QemuNbd::~QemuNbd() {
	sync();
	Process::exec("qemu-nbd -d /dev/nbd0");
}

QemuFlashDrive::QemuFlashDrive(const nlohmann::json& config_): FlashDrive(config_),
	qemu_connect(vir::connect_open("qemu:///system"))
{
	if (!is_defined()) {
		return;
	}

	for (auto& domain: qemu_connect.domains()) {
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		for (auto disk = devices.child("disk"); disk; disk = disk.next_sibling("disk")) {
			if (std::string(disk.attribute("device").value()) != "disk") {
				continue;
			}

			if (std::string(disk.child("source").attribute("file").value()) == img_path().generic_string()) {
				bool need_to_detach = false;
				auto metadata = config.first_child().child("metadata");
				if (metadata && metadata.child("testo:is_testo_related")) {
					need_to_detach = true;
				} else {
					std::string choice;
					std::cout << "Warning: Flash drive " << name() << " is plugged into user-defined vm " << domain.name() << std::endl;
					std::cout << "Would you like to unplug it? [Y/n]: ";
					std::getline(std::cin, choice);

					std::transform(choice.begin(), choice.end(), choice.begin(), ::toupper);

					if (!choice.length() || choice == "Y" || choice == "YES") {
						need_to_detach = true;
					} else {
						need_to_detach = false;
					}
				}

				if (need_to_detach) {
					std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

					if (domain.is_active()) {
						flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
					}
					domain.detach_device(disk, flags);
					return;
				}
			}
		}
	}
}

QemuFlashDrive::~QemuFlashDrive() {
	if (is_mounted()) {
		umount();
	}
}

bool QemuFlashDrive::is_defined() {
	try {
		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");
		for (auto& vol: pool.volumes()) {
			if (vol.name() == (id() + ".img")) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Is defined"));
	}
}

void QemuFlashDrive::create() {
	try {
		if (is_defined()) {
			undefine();
		}

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
		)", id(), config.at("size").get<uint32_t>(), img_path().generic_string()).c_str());

		auto volume = pool.volume_create_xml(xml_config, {VIR_STORAGE_VOL_CREATE_PREALLOC_METADATA});

		QemuNbd nbd_tmp(img_path());

		Process::exec("parted --script -a optimal /dev/nbd0 mklabel msdos mkpart primary ntfs 0% 100%");
		auto fs = config.at("fs").get<std::string>();
		if (fs == "ntfs") {
			Process::exec("mkfs." + fs + " -f /dev/nbd0p1");
		} else {
			Process::exec("mkfs." + fs + " /dev/nbd0p1");
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Creating flash drive"));
	}
}

void QemuFlashDrive::undefine() {
	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");

	auto vol = pool.storage_volume_lookup_by_name(id() + ".img");
	vol.erase({VIR_STORAGE_VOL_DELETE_NORMAL});
}

bool QemuFlashDrive::is_mounted() const {
	std::string query = "mountpoint -q \"" +env->flash_drives_mount_dir().generic_string() + "\"";
	return Process::is_succeeded(query.c_str());
}

void QemuFlashDrive::mount() {
	try {
		nbd.reset(new QemuNbd(img_path()));
		{
			coro::Timeout timeout{30s};
			coro::Timer timer;
			while (Process::is_failed("ls /dev/nbd0p1")) {
				timer.waitFor(1s);
			}
		}
		Process::exec("mount /dev/nbd0p1 " + env->flash_drives_mount_dir().generic_string());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive mount to host"));
	}
}

void QemuFlashDrive::umount() {
	try {
		sync();
		Process::exec("umount /dev/nbd0p1");
		nbd.reset(nullptr);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive umount from host"));
	}
}

bool QemuFlashDrive::has_snapshot(const std::string& snapshot) {
	try {
		std::string command = "qemu-img snapshot -l " + img_path().generic_string();
		std::string output = Process::exec(command);
		std::regex re("\\b" + snapshot + "\\b");
		std::smatch match;
		if (std::regex_search(output, match, re)) {
			return true;
		}
		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive umount from host"));
	}
}

void QemuFlashDrive::make_snapshot(const std::string& snapshot) {
	try {
		Process::exec("qemu-img snapshot -c " + snapshot + " " + img_path().generic_string());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive making snapshot"));
	}
}

void QemuFlashDrive::delete_snapshot(const std::string& snapshot) {
	try {
		Process::exec("qemu-img snapshot -d " + snapshot + " " + img_path().generic_string());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive deleting snapshot"));
	}
}

void QemuFlashDrive::rollback(const std::string& snapshot) {
	try {
		Process::exec("qemu-img snapshot -a " + snapshot + " " + img_path().generic_string());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive rolling back"));
	}
}

fs::path QemuFlashDrive::img_path() const {
	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");
	return pool.path() / (id() + ".img");
}

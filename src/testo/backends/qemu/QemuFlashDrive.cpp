
#include <coro/Timer.h>
#include <coro/Timeout.h>
#include <os/Process.hpp>
#include <pugixml/pugixml.hpp>
#include <guestfs/guestfs.hpp>
#include <fmt/format.h>
#include "QemuFlashDrive.hpp"
#include "QemuEnvironment.hpp"
#include <thread>
#include <fstream>
#include <regex>

using namespace std::chrono_literals;

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

		guestfs::Guestfs gfs(img_path());
		gfs.part_disk();
		gfs.mkfs(config.at("fs").get<std::string>());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Creating flash drive"));
	}
}

void QemuFlashDrive::undefine() {
	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");

	auto vol = pool.storage_volume_lookup_by_name(id() + ".img");
	vol.erase({VIR_STORAGE_VOL_DELETE_NORMAL});
}

void QemuFlashDrive::upload(const fs::path& from, const fs::path& to) {
	try {
		guestfs::Guestfs gfs(img_path());
		gfs.mount();

		if (to.has_parent_path()) {
			gfs.mkdir_p(to.parent_path());
		}
		gfs.upload(from, to);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Uploading from {} to {} on the flash drive {}", from.generic_string(), to.generic_string(), name())));
	}
}

void QemuFlashDrive::download(const fs::path& from, const fs::path& to) {
	try {
		guestfs::Guestfs gfs(img_path());
		gfs.mount();

		if (to.has_parent_path()) {
			fs::create_directories(to.parent_path());
		}
		gfs.download(from, to);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Downloading from {} to {} on the flash drive {}", from.generic_string(), to.generic_string(), name())));
	}
}

bool QemuFlashDrive::has_snapshot(const std::string& snapshot) {
	try {
		std::string command = "qemu-img snapshot -l " + img_path().generic_string();
		std::string output = os::Process::exec(command);
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
		os::Process::exec("qemu-img snapshot -c " + snapshot + " " + img_path().generic_string());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive making snapshot"));
	}
}

void QemuFlashDrive::delete_snapshot(const std::string& snapshot) {
	try {
		os::Process::exec("qemu-img snapshot -d " + snapshot + " " + img_path().generic_string());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive deleting snapshot"));
	}
}

void QemuFlashDrive::rollback(const std::string& snapshot) {
	try {
		os::Process::exec("qemu-img snapshot -a " + snapshot + " " + img_path().generic_string());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Flash drive rolling back"));
	}
}

fs::path QemuFlashDrive::img_path() const {
	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");
	return pool.path() / (id() + ".img");
}

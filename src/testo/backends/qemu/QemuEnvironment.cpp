
#include "QemuEnvironment.hpp"
#include "QemuVM.hpp"
#include "QemuFlashDrive.hpp"
#include "QemuNetwork.hpp"
#include <fmt/format.h>

QemuEnvironment::QemuEnvironment(): qemu_connect(vir::connect_open("qemu:///system")) {
}

void QemuEnvironment::prepare_storage_pool(const std::string& pool_name) {
	auto pool_dir = testo_dir() / pool_name;
	if (!fs::exists(pool_dir)) {
		if (!fs::create_directories(pool_dir)) {
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
	Environment::setup();

	prepare_storage_pool("testo-storage-pool");
	prepare_storage_pool("testo-flash-drives-pool");
}

std::shared_ptr<VM> QemuEnvironment::create_vm(const nlohmann::json& config) {
	return std::shared_ptr<VM>(new QemuVM(config));
}

std::shared_ptr<FlashDrive> QemuEnvironment::create_flash_drive(const nlohmann::json& config) {
	return std::shared_ptr<FlashDrive>(new QemuFlashDrive(config));
}

std::shared_ptr<Network> QemuEnvironment::create_network(const nlohmann::json& config) {
	return std::shared_ptr<Network>(new QemuNetwork(config));
}

void QemuEnvironment::validate_vm_config(const nlohmann::json& config) {

	if (config.count("disk")) {
		auto disks = config.at("disk");

		if (disks.size() > QemuVM::disk_targets.size() - 1) {
			throw std::runtime_error("Too many disks specified, maximum amount: " + std::to_string(QemuVM::disk_targets.size() - 1));
		}
	}

	if (config.count("nic")) {
		auto nics = config.at("nic");

		for (auto& nic: nics) {
			if (nic.count("adapter_type")) {
				std::string driver = nic.at("adapter_type").get<std::string>();
				if (driver != "ne2k_pci" &&
					driver != "i82551" &&
					driver != "i82557b" &&
					driver != "i82559er" &&
					driver != "rtl8139" &&
					driver != "e1000" &&
					driver != "pcnet" &&
					driver != "virtio" &&
					driver != "sungem")
				{
					throw std::runtime_error("NIC \"" +
						nic.at("name").get<std::string>() + "\" has unsupported adapter type: \"" + driver + "\"");
				}
			}
		}
	}

	if (config.count("video")) {
		auto videos = config.at("video");

		for (auto& video: videos) {
			auto video_model = video.value("adapter_type", video.value("qemu_mode", QemuVM::preferable_video_model(qemu_connect)));

			if ((video_model != "vmvga") &&
				(video_model != "vga") &&
				(video_model != "xen") &&
				(video_model != "virtio") &&
				(video_model != "qxl") &&
				(video_model != "cirrus"))
			{
				throw std::runtime_error("Video \"" +
					video.at("name").get<std::string>() + "\" has unsupported adapter type: \"" + video_model + "\"");
			}
		}
	}

}

void QemuEnvironment::validate_flash_drive_config(const nlohmann::json& config) {

}

void QemuEnvironment::validate_network_config(const nlohmann::json& config) {
	std::string id = config.at("prefix").get<std::string>() + config.at("name").get<std::string>();
	if (id.length() > 15) {
		throw std::runtime_error("Too long name for a network: " + id + ", please specifify 15 characters or less");
	}

}
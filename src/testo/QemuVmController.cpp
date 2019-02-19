
#include "QemuVmController.hpp"
#include "QemuFlashDriveController.hpp"

#include "Utils.hpp"
#include <fmt/format.h>
#include <regex>

QemuVmController::QemuVmController(const nlohmann::json& config): config(config),
	qemu_connect(vir::connect_open("qemu:///system"))
{

}


int QemuVmController::set_metadata(const nlohmann::json& metadata) {
	return 0;
}

int QemuVmController::set_metadata(const std::string& key, const std::string& value) {
	return 0;
}

std::vector<std::string> QemuVmController::keys() {
	return {};
}

bool QemuVmController::has_key(const std::string& key) {
	return true;
}

std::string QemuVmController::get_metadata(const std::string& key) {
	return "";
}

int QemuVmController::install() {
	if (is_defined()) {
		if (is_running()) {
			stop();
		}
		auto domain = qemu_connect.domain_lookup_by_name(name());
		for (auto& snapshot: domain.snapshots()) {
			snapshot.destroy();
		}

		auto xml = domain.dump_xml();

		domain.undefine();
		remove_disks(xml);
	}


	//now create disks
	//create_disks();
	return 0;
}

int QemuVmController::make_snapshot(const std::string& snapshot) {
	return 0;
}

std::set<std::string> QemuVmController::nics() const {
	return {};
}

int QemuVmController::set_snapshot_cksum(const std::string& snapshot, const std::string& cksum) {
	return 0;
}

std::string QemuVmController::get_snapshot_cksum(const std::string& snapshot) {
	return "";
}

int QemuVmController::rollback(const std::string& snapshot) {
	return 0;
}

int QemuVmController::press(const std::vector<std::string>& buttons) {
	return 0;
}

int QemuVmController::set_nic(const std::string& nic, bool is_enabled) {
	return 0;
}

int QemuVmController::set_link(const std::string& nic, bool is_connected) {
	return 0;
}

bool QemuVmController::is_plugged(std::shared_ptr<FlashDriveController> fd) {
	return true;
}

int QemuVmController::plug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	return 0;
}

int QemuVmController::unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	return 0;
}

void QemuVmController::unplug_all_flash_drives() {

}

int QemuVmController::plug_dvd(fs::path path) {
	return 0;
}

int QemuVmController::unplug_dvd() {
	return 0;
}

int QemuVmController::start() {
	return 0;
}

int QemuVmController::stop() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		domain.stop();
		return 0;
	}
	catch (const std::exception& error) {
		return -1;
	}
}

int QemuVmController::type(const std::string& text) {
	return 0;
}

int QemuVmController::wait(const std::string& text, const std::string& time) {
	return 0;
}

int QemuVmController::run(const fs::path& exe, std::vector<std::string> args) {
	return 0;
}

bool QemuVmController::has_snapshot(const std::string& snapshot) {
	auto domain = qemu_connect.domain_lookup_by_name(name());
	auto snapshots = domain.snapshots();
	for (auto& snap: snapshots) {
		if (snap.name() == snapshot) {
			return true;
		}
	}
	return false;
}

bool QemuVmController::is_defined() const {
	auto domains = qemu_connect.domains({VIR_CONNECT_LIST_DOMAINS_PERSISTENT});
	for (auto& domain: domains) {
		if (domain.name() == name()) {
			return true;
		}
	}
	return false;
}

bool QemuVmController::is_running() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		return domain.is_active();
	}
	catch (const std::exception& error) {
		std::cout << "Checking whether vm " << name() << " is running error : " << error << std::endl;
		return false;
	}
}

bool QemuVmController::is_additions_installed() {
	return true;
}


int QemuVmController::copy_to_guest(const fs::path& src, const fs::path& dst) {
	return 0;
}

int QemuVmController::remove_from_guest(const fs::path& obj) {
	return 0;
}

void QemuVmController::remove_disks(std::string xml) {
	std::string::size_type pos = 0; // Must initialize
	while (( pos = xml.find ("\n",pos)) != std::string::npos ) {
		xml.erase (pos, 1);
	}

	std::regex disks_regex("<disk.*?device='disk'.*?file='(.*?)'", std::regex::ECMAScript);
	auto disks_begin = std::sregex_iterator(xml.begin(), xml.end(), disks_regex);
	auto disks_end = std::sregex_iterator();

	for (auto i =  disks_begin; i != disks_end; ++i) {
		auto match = *i;
		fs::path disk_path(match[1].str());
		auto storage_volume = qemu_connect.storage_volume_lookup_by_path(disk_path);
		std::cout << "Erasing disk " << disk_path.generic_string() << std::endl;
		storage_volume.erase({VIR_STORAGE_VOL_DELETE_NORMAL});
	}
}

void QemuVmController::create_disks() {
	std::string storage_volume_config = fmt::format(R"(
		<volume type='file'>
		  <name>testing.img</name>
		  <source>
		  </source>
		  <capacity unit='bytes'>4294967296</capacity>
		  <allocation unit='bytes'>4335542272</allocation>
		  <physical unit='bytes'>7001014272</physical>
		  <target>
		    <path>/home/alex/testo/storage-pool/testing.img</path>
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
	)");
}


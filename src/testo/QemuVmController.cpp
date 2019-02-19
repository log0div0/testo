
#include "QemuVmController.hpp"
#include "QemuFlashDriveController.hpp"

#include "Utils.hpp"
#include <fmt/format.h>
#include <regex>

QemuVmController::QemuVmController(const nlohmann::json& config): config(config),
	qemu_connect(vir::connect_open("qemu:///session"))
{
	if (!config.count("name")) {
		throw std::runtime_error("Constructing QemuVmController error: field NAME is not specified");
	}

	if (!config.count("ram")) {
		throw std::runtime_error("Constructing QemuVmController error: field RAM is not specified");
	}

	if (!config.count("cpus")) {
		throw std::runtime_error("Constructing QemuVmController error: field CPUS is not specified");
	}

	if (!config.count("iso")) {
		throw std::runtime_error("Constructing QemuVmController error: field ISO is not specified");
	}

	fs::path iso_path(config.at("iso").get<std::string>());
	if (!fs::exists(iso_path)) {
		throw std::runtime_error(std::string("Constructing QemuVmController error: specified iso file does not exist: ")
			+ iso_path.generic_string());
	}

	if (!config.count("disk_size")) {
		throw std::runtime_error("Constructing QemuVmController error: field DISK SIZE is not specified");
	}
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
	create_disks();

	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-pool");
	fs::path volume_path = pool.path() / (name() + ".img");

	std::string xml_config = fmt::format(R"(
		<domain type='kvm'>
			<name>{}</name>
			<memory unit='KiB'>2097152</memory>
			<vcpu placement='static'>{}</vcpu>
			<resource>
				<partition>/machine</partition>
			</resource>
			<os>
				<type arch='x86_64' machine='ubuntu'>hvm</type>
				<boot dev='cdrom'/>
				<boot dev='hd'/>
			</os>
			<features>
				<acpi/>
				<apic/>
				<vmport state='off'/>
			</features>
			<cpu mode='host-model'>
				<model fallback='forbid'/>
			</cpu>
			<clock offset='utc'>
				<timer name='rtc' tickpolicy='catchup'/>
				<timer name='pit' tickpolicy='delay'/>
				<timer name='hpet' present='yes'/>
			</clock>
			<on_poweroff>destroy</on_poweroff>
			<on_reboot>restart</on_reboot>
			<on_crash>destroy</on_crash>
			<pm>
			</pm>
			<devices>
				<emulator>/usr/bin/kvm-spice</emulator>
				<disk type='file' device='disk'>
					<driver name='qemu' type='qcow2'/>
					<source file='{}'/>
					<target dev='vda' bus='virtio'/>
					<address type='pci' domain='0x0000' bus='0x00' slot='0x0a' function='0x0'/>
				</disk>
				<disk type='file' device='cdrom'>
					<driver name='qemu' type='raw'/>
					<source file='{}'/>
					<target dev='hda' bus='ide'/>
					<readonly/>
					<address type='drive' controller='0' bus='0' target='0' unit='0'/>
				</disk>
				<controller type='usb' index='0' model='ich9-ehci1'>
					<address type='pci' domain='0x0000' bus='0x00' slot='0x08' function='0x7'/>
				</controller>
				<controller type='usb' index='0' model='ich9-uhci1'>
					<master startport='0'/>
					<address type='pci' domain='0x0000' bus='0x00' slot='0x08' function='0x0' multifunction='on'/>
				</controller>
				<controller type='usb' index='0' model='ich9-uhci2'>
					<master startport='2'/>
					<address type='pci' domain='0x0000' bus='0x00' slot='0x08' function='0x1'/>
				</controller>
				<controller type='usb' index='0' model='ich9-uhci3'>
					<master startport='4'/>
					<address type='pci' domain='0x0000' bus='0x00' slot='0x08' function='0x2'/>
				</controller>
				<controller type='ide' index='0'>
					<address type='pci' domain='0x0000' bus='0x00' slot='0x01' function='0x1'/>
				</controller>
				<controller type='virtio-serial' index='0'>
					<address type='pci' domain='0x0000' bus='0x00' slot='0x09' function='0x0'/>
				</controller>
				<controller type='pci' index='0' model='pci-root'/>
				<serial type='pty'>
					<target type='isa-serial' port='0'>
						<model name='isa-serial'/>
					</target>
				</serial>
				<console type='pty'>
					<target type='serial' port='0'/>
				</console>
				<channel type='unix'>
					<target type='virtio' name='negotiator.0'/>
					<address type='virtio-serial' controller='0' bus='0' port='1'/>
				</channel>
				<channel type='spicevmc'>
					<target type='virtio' name='com.redhat.spice.0'/>
					<address type='virtio-serial' controller='0' bus='0' port='2'/>
				</channel>
				<input type='tablet' bus='usb'>
					<address type='usb' bus='0' port='1'/>
				</input>
				<input type='mouse' bus='ps2'/>
				<input type='keyboard' bus='ps2'/>
				<graphics type='spice' autoport='yes'>
					<listen type='address'/>
					<image compression='off'/>
				</graphics>
				<sound model='ich6'>
					<address type='pci' domain='0x0000' bus='0x00' slot='0x07' function='0x0'/>
				</sound>
				<video>
					<model type='qxl' ram='65536' vram='65536' vgamem='16384' heads='1' primary='yes'/>
					<address type='pci' domain='0x0000' bus='0x00' slot='0x02' function='0x0'/>
				</video>
				<redirdev bus='usb' type='spicevmc'>
					<address type='usb' bus='0' port='2'/>
				</redirdev>
				<redirdev bus='usb' type='spicevmc'>
					<address type='usb' bus='0' port='3'/>
				</redirdev>
				<memballoon model='virtio'>
					<address type='pci' domain='0x0000' bus='0x00' slot='0x0b' function='0x0'/>
				</memballoon>
			</devices>
		</domain>
	)", name(), config.at("cpus").get<uint32_t>(), volume_path.generic_string(), config.at("iso").get<std::string>());

	qemu_connect.domain_define_xml(xml_config);

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
	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-pool");
	std::string storage_volume_config = fmt::format(R"(
		<volume type='file'>
			<name>{}.img</name>
			<source>
			</source>
			<capacity unit='M'>{}</capacity>
			<target>
				<path>{}/{}.img</path>
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
	)", name(), config.at("disk_size").get<uint32_t>(), pool.path().generic_string(), name());

	auto volume = pool.volume_create_xml(storage_volume_config, {VIR_STORAGE_VOL_CREATE_PREALLOC_METADATA});
}


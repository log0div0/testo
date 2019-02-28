
#include "QemuVmController.hpp"
#include "QemuFlashDriveController.hpp"

#include "Utils.hpp"
#include "pugixml/pugixml.hpp"
#include <fmt/format.h>
#include <regex>

QemuVmController::QemuVmController(const nlohmann::json& config): config(config),
	qemu_connect(vir::connect_open("qemu:///system"))
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

	if (config.count("nic")) {
		auto nics = config.at("nic");
		for (auto& nic: nics) {
			if (!nic.count("attached_to")) {
				throw std::runtime_error("Constructing QemuVmController error: field attached_to is not specified for the nic " +
					nic.at("name").get<std::string>());
			}

			if (nic.at("attached_to").get<std::string>() == "internal") {
				if (!nic.count("network")) {
					throw std::runtime_error("Constructing QemuVmController error: nic " +
					nic.at("name").get<std::string>() + " has type internal, but field network is not specified");
				}
			}

			if (nic.count("mac")) {
				std::string mac = nic.at("mac").get<std::string>();
				if (!is_mac_correct(mac)) {
					throw std::runtime_error(std::string("Incorrect mac string: ") + mac);
				}
			}

			if (nic.at("attached_to").get<std::string>() == "nat") {
				if (nic.count("network")) {
					throw std::runtime_error("Constructing QemuVmController error: nic " +
					nic.at("name").get<std::string>() + " has type NAT, you must not specify field network");
				}
			}

			if (nic.count("adapter_type")) {
				//ne2k_pci,i82551,i82557b,i82559er,rtl8139,e1000,pcnet,virtio,sungem
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
					throw std::runtime_error("Constructing QemuVmController error: nic " +
						nic.at("name").get<std::string>() + " has unsupported adaptertype internal: " + driver);
				}
			}
		}

		for (uint32_t i = 0; i < nics.size(); i++) {
			for (uint32_t j = i + 1; j < nics.size(); j++) {
				if (nics[i].at("name") == nics[j].at("name")) {
					throw std::runtime_error("Constructing QemuVmController error: two identical NIC names: " +
						nics[i].at("name").get<std::string>());
				}
			}
		}
	}

	scancodes.insert({
		{"ESC", 1},
		{"ONE", 2},
		{"TWO", 3},
		{"THREE", 4},
		{"FOUR", 5},
		{"FIVE", 6},
		{"SIX", 7},
		{"SEVEN", 8},
		{"EIGHT", 9},
		{"NINE", 10},
		{"ZERO", 11},
		{"MINUS", 12},
		{"EQUALSIGN", 13},
		{"BACKSPACE", 14},
		{"TAB", 15},
		{"Q", 16},
		{"W", 17},
		{"E", 18},
		{"R", 19},
		{"T", 20},
		{"Y", 21},
		{"U", 22},
		{"I", 23},
		{"O", 24},
		{"P", 25},
		{"LEFTBRACE", 26},
		{"RIGHTBRACE", 27},
		{"ENTER", 28},
		{"LEFTCTRL", 29},
		{"A", 30},
		{"S", 31},
		{"D", 32},
		{"F", 33},
		{"G", 34},
		{"H", 35},
		{"J", 36},
		{"K", 37},
		{"L", 38},
		{"SEMICOLON", 39},
		{"APOSTROPHE", 40},
		{"GRAVE", 41},
		{"LEFTSHIFT", 42},
		{"BACKSLASH", 43},
		{"Z", 44},
		{"X", 45},
		{"C", 46},
		{"V", 47},
		{"B", 48},
		{"N", 49},
		{"M", 50},
		{"COMMA", 51},
		{"DOT", 52},
		{"SLASH", 53},
		{"RIGHTSHIFT", 54},
		{"LEFTALT", 56},
		{"SPACE", 57},
		{"CAPSLOCK", 58},
		{"NUMLOCK", 69},
		{"SCROLLLOCK", 70},
		{"RIGHTCTRL", 97},
		{"RIGHTALT", 100},
		{"HOME", 102},
		{"UP", 103},
		{"PAGEUP", 104},
		{"LEFT", 105},
		{"RIGHT", 106},
		{"END", 107},
		{"DOWN", 108},
		{"PAGEDOWN", 109},
		{"INSERT", 110},
		{"DELETE", 111},
		{"SCROLLUP", 177},
		{"SCROLLDOWN", 178},
	});

	charmap.insert({
		{'0', {"0"}},
		{'1', {"1"}},
		{'2', {"2"}},
		{'3', {"3"}},
		{'4', {"4"}},
		{'5', {"5"}},
		{'6', {"6"}},
		{'7', {"7"}},
		{'8', {"8"}},
		{'9', {"9"}},
		{')', {"LEFTSHIFT", "0"}},
		{'!', {"LEFTSHIFT", "1"}},
		{'@', {"LEFTSHIFT", "2"}},
		{'#', {"LEFTSHIFT", "3"}},
		{'$', {"LEFTSHIFT", "4"}},
		{'%', {"LEFTSHIFT", "5"}},
		{'^', {"LEFTSHIFT", "6"}},
		{'&', {"LEFTSHIFT", "7"}},
		{'*', {"LEFTSHIFT", "8"}},
		{'(', {"LEFTSHIFT", "9"}},
		{'a', {"A"}},
		{'b', {"B"}},
		{'c', {"C"}},
		{'d', {"D"}},
		{'e', {"E"}},
		{'f', {"F"}},
		{'g', {"G"}},
		{'h', {"H"}},
		{'i', {"I"}},
		{'j', {"J"}},
		{'k', {"K"}},
		{'l', {"L"}},
		{'m', {"M"}},
		{'n', {"N"}},
		{'o', {"O"}},
		{'p', {"P"}},
		{'q', {"Q"}},
		{'r', {"R"}},
		{'s', {"S"}},
		{'t', {"T"}},
		{'u', {"U"}},
		{'v', {"V"}},
		{'w', {"W"}},
		{'x', {"X"}},
		{'y', {"Y"}},
		{'z', {"Z"}},
		{'A', {"LEFTSHIFT", "A"}},
		{'B', {"LEFTSHIFT", "B"}},
		{'C', {"LEFTSHIFT", "C"}},
		{'D', {"LEFTSHIFT", "D"}},
		{'E', {"LEFTSHIFT", "E"}},
		{'F', {"LEFTSHIFT", "F"}},
		{'G', {"LEFTSHIFT", "G"}},
		{'H', {"LEFTSHIFT", "H"}},
		{'I', {"LEFTSHIFT", "I"}},
		{'J', {"LEFTSHIFT", "J"}},
		{'K', {"LEFTSHIFT", "K"}},
		{'L', {"LEFTSHIFT", "L"}},
		{'M', {"LEFTSHIFT", "M"}},
		{'N', {"LEFTSHIFT", "N"}},
		{'O', {"LEFTSHIFT", "O"}},
		{'P', {"LEFTSHIFT", "P"}},
		{'Q', {"LEFTSHIFT", "Q"}},
		{'R', {"LEFTSHIFT", "R"}},
		{'S', {"LEFTSHIFT", "S"}},
		{'T', {"LEFTSHIFT", "T"}},
		{'U', {"LEFTSHIFT", "U"}},
		{'V', {"LEFTSHIFT", "V"}},
		{'W', {"LEFTSHIFT", "W"}},
		{'X', {"LEFTSHIFT", "X"}},
		{'Y', {"LEFTSHIFT", "Y"}},
		{'Z', {"LEFTSHIFT", "Z"}},
		{'-', {"MINUS"}},
		{'_', {"LEFTSHIFT", "MINUS"}},
		{'=', {"EQUALSIGN"}},
		{'+', {"LEFTSHIFT", "EQUALSIGN"}},
		{'\'', {"APOSTROPHE"}},
		{'\"', {"LEFTSHIFT", "APOSTROPHE"}},
		{'\\', {"BACKSLASH"}},
		{'|', {"LEFTSHIFT", "BACKSLASH"}},
		{',', {"COMMA"}},
		{'<', {"LEFTSHIFT", "COMMA"}},
		{'.', {"DOT"}},
		{'>', {"LEFTSHIFT", "DOT"}},
		{'/', {"SLASH"}},
		{'?', {"LEFTSHIFT", "SLASH"}},
		{';', {"SEMICOLON"}},
		{':', {"LEFTSHIFT", "SEMICOLON"}},
		{'[', {"LEFTBRACE"}},
		{'{', {"LEFTSHIFT", "LEFTBRACE"}},
		{']', {"RIGHTBRACE"}},
		{'}', {"LEFTSHIFT", "RIGHTBRACE"}},
		{'`', {"GRAVE"}},
		{'~', {"LEFTSHIFT", "GRAVE"}},
		{' ', {"SPACE"}}
	});

	prepare_networks();
}


int QemuVmController::set_metadata(const nlohmann::json& metadata) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		for (auto key_value = metadata.begin(); key_value != metadata.end(); ++key_value) {
			if (set_metadata(key_value.key(), key_value.value()) < 0) {
				std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
			}
		}
		return 0;
	}
	catch (const std::exception& error) {
		std::cout << "Setting metadata on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int QemuVmController::set_metadata(const std::string& key, const std::string& value) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		std::vector flags = {VIR_DOMAIN_AFFECT_CURRENT, VIR_DOMAIN_AFFECT_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_AFFECT_LIVE);
		}

		std::string metadata_to_set = value.length() ? fmt::format("<{} value='{}'/>", key, value) : "";

		domain.set_metadata(VIR_DOMAIN_METADATA_ELEMENT,
			metadata_to_set,
			"testo",
			fmt::format("vm_metadata/{}", key),
			flags);
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Setting metadata with key " << key << " on vm " << name() << " error : " << error << std::endl;
		return -1;
	}
}

std::vector<std::string> QemuVmController::keys() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		return metadata_keys(domain.dump_xml());
	}
	catch (const std::exception& error) {
		std::cout << "Getting metadata keys on vm " << name() << ": " << error << std::endl;
		return {};
	}
}

bool QemuVmController::has_key(const std::string& key) {
	try {
		auto config = qemu_connect.domain_lookup_by_name(name()).dump_xml_new();
		auto found = config.select_node(fmt::format("//*[namespace-uri() = \"vm_metadata/{}\"]", key).c_str());
		return !found.node().empty();
	} catch (const std::exception& error) {
		std::cout << "Checking metadata with key " << key << " on vm " << name() << " error : " << error << std::endl;
		return false;
	}
}

std::string QemuVmController::get_metadata(const std::string& key) {
	try {
		auto config = qemu_connect.domain_lookup_by_name(name()).dump_xml_new();
		auto found = config.select_node(fmt::format("//*[namespace-uri() = \"vm_metadata/{}\"]", key).c_str()).node();
		if (found.empty()) {
			throw std::runtime_error("Requested key is not present in vm metadata");
		}

		return found.attribute("value").value();

	} catch (const std::exception& error) {
		std::cout << "Getting metadata with key " << key << " on vm " << name() << " error : " << error << std::endl;
		return "";
	}
}

int QemuVmController::install() {
	try {
		if (is_defined()) {
			if (is_running()) {
				if (stop()) {
					std::throw_with_nested(__PRETTY_FUNCTION__);
				}
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
				<metadata>
					<testo:login xmlns:testo="vm_metadata/login" value='root'/>
					<testo:password xmlns:testo="vm_metadata/password" value='1111'/>
				</metadata>
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
					</disk>
					<disk type='file' device='cdrom'>
						<driver name='qemu' type='raw'/>
						<source file='{}'/>
						<target dev='hda' bus='ide'/>
						<readonly/>
					</disk>
					<controller type='usb' index='0' model='ich9-ehci1'>
					</controller>
					<controller type='usb' index='0' model='ich9-uhci1'>
					</controller>
					<controller type='usb' index='0' model='ich9-uhci2'>
					</controller>
					<controller type='usb' index='0' model='ich9-uhci3'>
					</controller>
					<controller type='ide' index='0'>
					</controller>
					<controller type='virtio-serial' index='0'>
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
					</channel>
					<channel type='spicevmc'>
						<target type='virtio' name='com.redhat.spice.0'/>
					</channel>
					<input type='tablet' bus='usb'>
					</input>
					<input type='mouse' bus='ps2'/>
					<input type='keyboard' bus='ps2'/>
					<graphics type='spice' autoport='yes'>
						<listen type='address'/>
						<image compression='off'/>
					</graphics>
					<sound model='ich6'>
					</sound>
					<video>
						<model type='qxl' ram='65536' vram='65536' vgamem='16384' heads='1' primary='yes'/>
					</video>
					<redirdev bus='usb' type='spicevmc'>
					</redirdev>
					<redirdev bus='usb' type='spicevmc'>
					</redirdev>
					<memballoon model='virtio'>
					</memballoon>
		)", name(), config.at("cpus").get<uint32_t>(), volume_path.generic_string(), config.at("iso").get<std::string>());

		uint32_t nic_count = 0;

		if (config.count("nic")) {
			auto nics = config.at("nic");
			for (auto& nic: nics) {
				std::string source_network("testo-");

				if (nic.at("attached_to").get<std::string>() == "internal") {
					source_network += nic.at("network").get<std::string>();
				}

				if (nic.at("attached_to").get<std::string>() == "nat") {
					source_network += "nat";
				}

				xml_config += fmt::format(R"(
					<interface type='network'>
						<source network='{}'/>
				)", source_network);

				if (nic.count("mac")) {
					xml_config += fmt::format("\n<mac address='{}'/>", nic.at("mac").get<std::string>());
				}

				if (nic.count("adapter_type")) {
					xml_config += fmt::format("\n<model type='{}'/>", nic.at("adapter_type").get<std::string>());
				}

				xml_config += fmt::format("\n</interface>");

				nic_count++;
			}
		}

		xml_config += "\n </devices> \n </domain>";
		auto domain = qemu_connect.domain_define_xml(xml_config);

		if (config.count("metadata")) {
			set_metadata(config.at("metadata"));
		}

		auto config_str = config.dump();

		set_metadata("vm_config", config_str);
		set_metadata("vm_nic_count", std::to_string(nic_count));
		set_metadata("vm_name", name());

		domain.start();
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Performing install on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int QemuVmController::make_snapshot(const std::string& snapshot, const std::string& cksum) {
	std::string xml_config = fmt::format(R"(
		<domainsnapshot>
			<name>{}</name>
			<description>{}</description>
		</domainsnapshot>
		)", snapshot, cksum);

	auto domain = qemu_connect.domain_lookup_by_name(name());
	domain.snapshot_create_xml(xml_config);
	return 0;
}

std::set<std::string> QemuVmController::nics() const {
	return {};
}

std::string QemuVmController::get_snapshot_cksum(const std::string& snapshot) {
	try {
		auto xml = qemu_connect.domain_lookup_by_name(name()).snapshot_lookup_by_name(snapshot).dump_xml();

		remove_newlines(xml);
		std::regex description_regex(".*?<description>(.*?)<\\/description>.*", std::regex::ECMAScript);
		std::smatch match;

		if (std::regex_match(xml, match, description_regex)) {
			std::string value = match[1].str();
			return value;
		} else {
			throw std::runtime_error("Description is not present in snapshot metadata");
		}

		return "";
	}
	catch (const std::exception& error) {
		std::cout << "getting snapshot cksum on vm " << name() << ": " << error << std::endl;
		return "";
	}
}

int QemuVmController::rollback(const std::string& snapshot) {
	try {

		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto snap = domain.snapshot_lookup_by_name(snapshot);
		domain.revert_to_snapshot(snap);

		//Now let's take care of possible additional metadata keys

		auto new_metadata_keys = metadata_keys(domain.dump_xml());
		auto old_metadata_keys = metadata_keys(snap.dump_xml());

		std::sort(new_metadata_keys.begin(), new_metadata_keys.end());
		std::sort(old_metadata_keys.begin(), old_metadata_keys.end());

		std::vector<std::string> difference;
		std::set_difference(new_metadata_keys.begin(), new_metadata_keys.end(),
			old_metadata_keys.begin(), old_metadata_keys.end(), std::back_inserter(difference));

		for (auto& key: difference) {
			set_metadata(key, "");
		}

		return 0;
	} catch (const std::exception& error) {
		std::cout << "Performing rollback on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int QemuVmController::press(const std::vector<std::string>& buttons) {
	try {
		std::vector<uint32_t> keycodes;
		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);
			keycodes.push_back(scancodes[button]);
		}
		qemu_connect.domain_lookup_by_name(name()).send_keys(VIR_KEYCODE_SET_LINUX, 0, keycodes);
		return 0;
	}
	catch (const std::exception& error) {
		std::cout << "Pressing buttons on vm " << name() << " error : " << error << std::endl;
		return -1;
	}
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
	try {
	/*	auto domain = qemu_connect.domain_lookup_by_name(name());
		auto xml = domain.dump_xml();
		remove_newlines(xml);

		std::regex description_regex(".*?<description>(.*?)<\\/description>.*", std::regex::ECMAScript);
		std::smatch match;

		if (std::regex_match(xml, match, description_regex)) {
			std::string value = match[1].str();
			return value;
		} else {
			throw std::runtime_error("Description is not present in snapshot metadata");
		}*/

		return 0;

	} catch (const std::string& error) {
		std::cout << "Unplugging dvd from vm " << name() << ": " << error << std::endl;
		return -1;
	}

}

int QemuVmController::start() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		domain.start();
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Starting vm " << name() << ": " << error << std::endl;
		return -1;
	}
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
	try {
		for (auto c: text) {
			auto buttons = charmap.find(c);
			if (buttons == charmap.end()) {
				throw std::runtime_error("Unknown character to type");
			}

			press(buttons->second);
		}
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Typing on vm " << name() << ": " << error << std::endl;
		return -1;
	}

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

void QemuVmController::prepare_networks() {
	if (config.count("nic")) {
		auto nics = config.at("nic");
		for (auto& nic: nics) {
			std::string network_to_lookup;
			if (nic.at("attached_to").get<std::string>() == "nat") {
				network_to_lookup = "testo-nat";
			}

			if (nic.at("attached_to").get<std::string>() == "internal") {
				network_to_lookup = std::string("testo-") + nic.at("network").get<std::string>();
			}

			bool found = false;
			for (auto& network: qemu_connect.networks()) {
				if (network.name() == network_to_lookup) {
					if (!network.is_active()) {
						network.start();
					}
					found = true;
					break;
				}
			}

			if (!found) {
				std::string xml_config = fmt::format(R"(
					<network>
						<name>{}</name>
						<bridge name="{}"/>
				)", network_to_lookup, network_to_lookup);

				if (network_to_lookup == "testo-nat") {
					xml_config += fmt::format(R"(
						<forward mode='nat'>
							<nat>
								<port start='1024' end='65535'/>
							</nat>
						</forward>
						<ip address='192.168.156.1' netmask='255.255.255.0'>
							<dhcp>
								<range start='192.168.156.2' end='192.168.156.254'/>
							</dhcp>
						</ip>
					)");
				}

				xml_config += "\n</network>";
				auto network = qemu_connect.network_define_xml(xml_config);
				network.start();
			}
		}
	}
}


void QemuVmController::remove_disks(std::string xml) {
	remove_newlines(xml);
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

std::vector<std::string> QemuVmController::metadata_keys(std::string xml) const {
	std::vector<std::string> result;
	remove_newlines(xml);

	std::regex keys_regex("<testo:(.*?)\\ .*?/>", std::regex::ECMAScript);
	auto keys_begin = std::sregex_iterator(xml.begin(), xml.end(), keys_regex);
	auto keys_end = std::sregex_iterator();

	for (auto i =  keys_begin; i != keys_end; ++i) {
		auto match = *i;
		result.push_back(match[1].str());
	}

	return result;
}


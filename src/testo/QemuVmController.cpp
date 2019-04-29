
#include "QemuVmController.hpp"
#include "QemuFlashDriveController.hpp"
#include "Negotiator.hpp"

#include "Utils.hpp"
#include <fmt/format.h>
#include <thread>

#include <stb_image.h>

QemuVmController::QemuVmController(const nlohmann::json& config): config(config),
	qemu_connect(vir::connect_open("qemu:///system")), screenshot_buffer(10'000'000)
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
		{'0', {"ZERO"}},
		{'1', {"ONE"}},
		{'2', {"TWO"}},
		{'3', {"THREE"}},
		{'4', {"FOUR"}},
		{'5', {"FIVE"}},
		{'6', {"SIX"}},
		{'7', {"SEVEN"}},
		{'8', {"EIGHT"}},
		{'9', {"NINE"}},
		{')', {"LEFTSHIFT", "ZERO"}},
		{'!', {"LEFTSHIFT", "ONE"}},
		{'@', {"LEFTSHIFT", "TWO"}},
		{'#', {"LEFTSHIFT", "THREE"}},
		{'$', {"LEFTSHIFT", "FOUR"}},
		{'%', {"LEFTSHIFT", "FIVE"}},
		{'^', {"LEFTSHIFT", "SIX"}},
		{'&', {"LEFTSHIFT", "SEVEN"}},
		{'*', {"LEFTSHIFT", "EIGHT"}},
		{'(', {"LEFTSHIFT", "NINE"}},
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
		{'\n', {"ENTER"}},
		{'\t', {"TAB"}},
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

QemuVmController::~QemuVmController() {
	if (!is_defined()) {
		remove_disk();
	}
}

void QemuVmController::set_metadata(const nlohmann::json& metadata) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		for (auto key_value = metadata.begin(); key_value != metadata.end(); ++key_value) {
			set_metadata(key_value.key(), key_value.value());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Setting json metadata on vm "));
	}
}

void QemuVmController::set_metadata(const std::string& key, const std::string& value) {
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
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting metadata with key {}", key)));
	}
}

std::vector<std::string> QemuVmController::keys() {
	try {
		std::vector<std::string> result;
		auto config = qemu_connect.domain_lookup_by_name(name()).dump_xml();
		auto metadata = config.first_child().child("metadata");
		for (auto it = metadata.begin(); it != metadata.end(); ++it) {
			std::string value = it->first_attribute().value();
			result.push_back(value.substr(strlen("vm_metadata/")));
		}

		return result;
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting metadata keys")));
	}
}

std::vector<std::string> QemuVmController::keys(vir::Snapshot& snapshot) {
	try {
		std::vector<std::string> result;
		auto xml = snapshot.dump_xml();
		auto metadata = xml.first_child().child("domain").child("metadata");
		for (auto it = metadata.begin(); it != metadata.end(); ++it) {
			std::string value = it->first_attribute().value();
			result.push_back(value.substr(strlen("vm_metadata/")));
		}

		return result;

	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting metadata keys")));
	}
}

bool QemuVmController::has_key(const std::string& key) {
	try {
		auto config = qemu_connect.domain_lookup_by_name(name()).dump_xml();
		auto found = config.select_node(fmt::format("//*[namespace-uri() = \"vm_metadata/{}\"]", key).c_str());
		return !found.node().empty();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking metadata with key {}", key)));
	}
}

std::string QemuVmController::get_metadata(const std::string& key) {
	try {
		auto config = qemu_connect.domain_lookup_by_name(name()).dump_xml();
		auto found = config.select_node(fmt::format("//*[namespace-uri() = \"vm_metadata/{}\"]", key).c_str()).node();
		if (found.empty()) {
			throw std::runtime_error("Requested key is not present in vm metadata");
		}

		return found.attribute("value").value();

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting metadata with key {}", key)));
	}
}

void QemuVmController::install() {
	try {
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
		}

		remove_disk();

		//now create disks
		create_disk();

		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");
		fs::path volume_path = pool.path() / (name() + ".img");

		std::string string_config = fmt::format(R"(
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

				string_config += fmt::format(R"(
					<interface type='network'>
						<source network='{}'/>
				)", source_network);

				if (nic.count("mac")) {
					string_config += fmt::format("\n<mac address='{}'/>", nic.at("mac").get<std::string>());
				}

				if (nic.count("adapter_type")) {
					string_config += fmt::format("\n<model type='{}'/>", nic.at("adapter_type").get<std::string>());
				}

				//libvirt suggests that everything you do in aliases must be prefixed with "ua-nic-"
				std::string nic_name = std::string("ua-nic-");
				nic_name += nic.at("name").get<std::string>();
				string_config += fmt::format("\n<link state='up'/>");
				string_config += fmt::format("\n<alias name='{}'/>", nic_name);
				string_config += fmt::format("\n</interface>");

				nic_count++;
			}
		}

		string_config += "\n </devices> \n </domain>";

		pugi::xml_document xml_config;
		xml_config.load_string(string_config.c_str());
		auto domain = qemu_connect.domain_define_xml(xml_config);

		if (config.count("metadata")) {
			set_metadata(config.at("metadata"));
		}

		auto config_str = config.dump();

		set_metadata("vm_config", config_str);
		set_metadata("vm_nic_count", std::to_string(nic_count));
		set_metadata("vm_name", name());
		set_metadata("dvd_signature", file_signature(config.at("iso").get<std::string>()));

		domain.start();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Performing install")));
	}
}

void QemuVmController::make_snapshot(const std::string& snapshot, const std::string& cksum) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());

		if (has_snapshot(snapshot)) {
			auto vir_snapshot = domain.snapshot_lookup_by_name(snapshot);
			delete_snapshot_with_children(vir_snapshot);
		}

		pugi::xml_document xml_config;
		xml_config.load_string(fmt::format(R"(
			<domainsnapshot>
				<name>{}</name>
				<description>{}</description>
			</domainsnapshot>
			)", snapshot, cksum).c_str());


		domain.snapshot_create_xml(xml_config);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Taking snapshot")));
	}

}

std::set<std::string> QemuVmController::nics() const {
	std::set<std::string> result;

	for (auto& nic: config.at("nic")) {
		result.insert(nic.at("name").get<std::string>());
	}
	return result;
}

std::string QemuVmController::get_snapshot_cksum(const std::string& snapshot) {
	try {
		auto config = qemu_connect.domain_lookup_by_name(name()).snapshot_lookup_by_name(snapshot).dump_xml();
		auto description = config.first_child().child("description");
		return description.text().get();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("getting snapshot cksum error"));
	}
}

void QemuVmController::rollback(const std::string& snapshot) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto snap = domain.snapshot_lookup_by_name(snapshot);

		//Now let's take care of possible additional metadata keys

		auto new_metadata_keys = keys();
		auto old_metadata_keys = keys(snap);

		std::sort(new_metadata_keys.begin(), new_metadata_keys.end());
		std::sort(old_metadata_keys.begin(), old_metadata_keys.end());

		std::vector<std::string> difference;
		std::set_difference(new_metadata_keys.begin(), new_metadata_keys.end(),
			old_metadata_keys.begin(), old_metadata_keys.end(), std::back_inserter(difference));

		for (auto& key: difference) {
			set_metadata(key, "");
		}

		//Now let's take care of possible dvd discontingency
		std::string current_dvd = get_dvd_path();
		std::string snapshot_dvd = get_dvd_path(snap);

		if (current_dvd != snapshot_dvd) {
			//Possible variations:
			//If we have something plugged - let's unplug it
			if (current_dvd.length()) {
				unplug_dvd();
			}

			if (snapshot_dvd.length()) {
				plug_dvd(snapshot_dvd);
			}
		}

		//nics contingency
		for (auto& nic: config.at("nic")) {

			std::string nic_name = nic.at("name").get<std::string>();
			auto currently_plugged = is_nic_plugged(nic_name);
			auto snapshot_plugged = is_nic_plugged(snap, nic_name);
			if (currently_plugged != snapshot_plugged) {
				if (is_running()) {
					stop();
				}

				set_nic(nic_name, snapshot_plugged);
			}
		}

		//links contingency
		for (auto& nic: nics()) {
			auto currently_plugged = is_link_plugged(nic);
			auto snapshot_plugged = is_link_plugged(snap, nic);
			if (currently_plugged != snapshot_plugged) {
				set_link(nic, snapshot_plugged);
			}
		}

		std::string flash_attached = get_flash_img();
		if (flash_attached.length()) {
			detach_flash_drive();
		}

		domain.revert_to_snapshot(snap);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Performing rollback error"));
	}
}

void QemuVmController::press(const std::vector<std::string>& buttons) {
	try {
		std::vector<uint32_t> keycodes;
		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);
			keycodes.push_back(scancodes[button]);
		}
		qemu_connect.domain_lookup_by_name(name()).send_keys(VIR_KEYCODE_SET_LINUX, 0, keycodes);
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Pressing buttons error"));
	}
}

bool QemuVmController::is_nic_plugged(const std::string& nic) const {
	try {
		auto nic_name = std::string("ua-nic-") + nic;
		auto config = qemu_connect.domain_lookup_by_name(name()).dump_xml();
		auto devices = config.first_child().child("devices");

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			if (std::string(nic_node.child("alias").attribute("name").value()) == nic_name) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking if nic {} is plugged", nic)));
	}
}

bool QemuVmController::is_nic_plugged(vir::Snapshot& snapshot, const std::string& nic) {
	try {
		auto nic_name = std::string("ua-nic-") + nic;
		auto config = snapshot.dump_xml();
		auto devices = config.first_child().child("domain").child("devices");

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			if (std::string(nic_node.child("alias").attribute("name").value()) == nic_name) {
				return true;
			}
		}

		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking if nic {} is plugged from snapshot", nic)));
	}

}

void QemuVmController::attach_nic(const std::string& nic) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());

		std::string string_config;

		for (auto& nic_json: config.at("nic")) {
			if (nic_json.at("name") == nic) {
				std::string source_network("testo-");

				if (nic_json.at("attached_to").get<std::string>() == "internal") {
					source_network += nic_json.at("network").get<std::string>();
				}

				if (nic_json.at("attached_to").get<std::string>() == "nat") {
					source_network += "nat";
				}

				string_config = fmt::format(R"(
					<interface type='network'>
						<source network='{}'/>
				)", source_network);

				if (nic_json.count("mac")) {
					string_config += fmt::format("\n<mac address='{}'/>", nic_json.at("mac").get<std::string>());
				}

				if (nic_json.count("adapter_type")) {
					string_config += fmt::format("\n<model type='{}'/>", nic_json.at("adapter_type").get<std::string>());
				}

				//libvirt suggests that everything you do in aliases must be prefixed with "ua-nic-"
				std::string nic_name = std::string("ua-nic-");
				nic_name += nic_json.at("name").get<std::string>();
				string_config += fmt::format("\n<link state='up'/>");
				string_config += fmt::format("\n<alias name='{}'/>", nic_name);
				string_config += fmt::format("\n</interface>");

				break;
			}
		}

		//TODO: check if CURRENT is enough
		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document nic_config;
		nic_config.load_string(string_config.c_str());

		domain.attach_device(nic_config, flags);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Attaching nic {}", nic)));
	}

}

void QemuVmController::detach_nic(const std::string& nic) {
	try {
		auto nic_name = std::string("ua-nic-") + nic;
		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		//TODO: check if CURRENT is enough
		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			if (std::string(nic_node.child("alias").attribute("name").value()) == nic_name) {
				domain.detach_device(nic_node, flags);
				return;
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Detaching nic {}", nic)));
	}
}

void QemuVmController::set_nic(const std::string& nic, bool is_enabled) {
	try {
		if (is_enabled) {
			attach_nic(nic);
		} else {
			detach_nic(nic);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting nic {}", nic)));
	}
}

bool QemuVmController::is_link_plugged(const pugi::xml_node& devices, const std::string& nic) const {
	try {
		std::string nic_name = std::string("ua-nic-") + nic;

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			if (std::string(nic_node.child("alias").attribute("name").value()) == nic_name) {
				if (nic_node.child("link").empty()) {
					return false;
				}

				std::string state = nic_node.child("link").attribute("state").value();

				if (state == "up") {
					return true;
				} else if (state == "down") {
					return false;
				} else {
					throw std::runtime_error("Unknown link state");
				}
			}
		}
		return false;
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(fmt::format("Checking if nic {} is plugged", nic)));
	}
}

bool QemuVmController::is_link_plugged(vir::Snapshot& snapshot, const std::string& nic) {
	try {
		auto config = snapshot.dump_xml();
		return is_link_plugged(config.first_child().child("domain").child("devices"), nic);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(fmt::format("Checking if nic {} is plugged from snapshot", nic)));
	}

}

bool QemuVmController::is_link_plugged(const std::string& nic) const {
	try {
		auto config = qemu_connect.domain_lookup_by_name(name()).dump_xml();
		return is_link_plugged(config.first_child().child("devices"), nic);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking link status on nic {}", nic)));
	}
}

void QemuVmController::set_link(const std::string& nic, bool is_connected) {
	try {
		std::string nic_name = std::string("ua-nic-") + nic;
		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");
		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			if (std::string(nic_node.child("alias").attribute("name").value()) == nic_name) {
				if (is_connected) { //connect link
					//if we have set link attribute - just change state to up
					if (!nic_node.child("link").empty()) {
						nic_node.child("link").attribute("state").set_value("up");
					}
				} else { //disconnect link
					//if we have set link attribute - set it to down
					if (!nic_node.child("link").empty()) {
						nic_node.child("link").attribute("state").set_value("down");
					} else {
						auto link = nic_node.insert_child_before("link", nic_node.child("alias"));
						link.append_attribute("state") = "down";
					}
				}

				std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

				if (domain.is_active()) {
					flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
				}

				domain.update_device(nic_node, flags);
				break;
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting link status on nic {}", nic)));
	}
}

std::string QemuVmController::get_flash_img() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		std::string result = "";

		for (auto disk = devices.child("disk"); disk; disk = disk.next_sibling("disk")) {
			if (std::string(disk.attribute("device").value()) != "disk") {
				continue;
			}

			if (std::string(disk.child("target").attribute("dev").value()) == "vdb") {
				result = disk.child("source").attribute("file").value();
			}
		}

		return result;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Getting flash image"));
	}
}

bool QemuVmController::is_flash_plugged(std::shared_ptr<FlashDriveController> fd) {
	try {
		return get_flash_img().length();
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking if flash drive {} is pluged", fd->name())));
	}
}

void QemuVmController::attach_flash_drive(const std::string& img_path) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());

		std::string string_config = fmt::format(R"(
			<disk type='file'>
				<driver name='qemu' type='qcow2'/>
				<source file='{}'/>
				<target dev='vdb' bus='virtio'/>
			</disk>
			)", img_path);

		//we just need to create new device
		//TODO: check if CURRENT is enough
		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_CURRENT};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document disk_config;
		disk_config.load_string(string_config.c_str());

		domain.attach_device(disk_config, flags);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Attaching flash drive {}", img_path)));
	}

}

void QemuVmController::plug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	try {
		attach_flash_drive(fd->img_path());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Plugging flash drive {}", fd->name())));
	}
}

void QemuVmController::detach_flash_drive() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		//TODO: check if CURRENT is enough
		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		for (auto disk = devices.child("disk"); disk; disk = disk.next_sibling("disk")) {
			if (std::string(disk.attribute("device").value()) != "disk") {
				continue;
			}

			if (std::string(disk.child("target").attribute("dev").value()) == "vdb") {
				domain.detach_device(disk, flags);
				break;
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Detaching flash drive"));
	}
}

//for now it's just only one flash drive possible
void QemuVmController::unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	try {
		detach_flash_drive();
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Unplugging flash drive {}", fd->name())));
	}
}


bool QemuVmController::is_dvd_plugged() const {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");
		return !bool(cdrom.child("source").empty());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Checking if dvd is plugged"));
	}
}

std::string QemuVmController::get_dvd_path() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");
		if (cdrom.child("source").empty()) {
			return "";
		}
		return cdrom.child("source").attribute("file").value();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Getting dvd path"));
	}
}


std::string QemuVmController::get_dvd_path(vir::Snapshot& snap) {
	try {
		auto config = snap.dump_xml();
		auto cdrom = config.first_child().child("domain").child("devices").find_child_by_attribute("device", "cdrom");
		if (cdrom.child("source").empty()) {
			return "";
		}
		return cdrom.child("source").attribute("file").value();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Getting dvd path from snapshot"));
	}
}

void QemuVmController::plug_dvd(fs::path path) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");

		if (!cdrom.child("source").empty()) {
			throw std::runtime_error("Some dvd is already plugged in");
		}

		auto source = cdrom.insert_child_after("source", cdrom.child("driver"));
		source.append_attribute("file") = path.generic_string().c_str();

		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_CURRENT};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}
		domain.update_device(cdrom, flags);
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("plugging dvd {}", path.generic_string())));
	}
}

void QemuVmController::unplug_dvd() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");

		if (cdrom.child("source").empty()) {
			throw std::runtime_error("Dvd is already unplugged");
		}

		cdrom.remove_child("source");

		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		domain.update_device(cdrom, flags);
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error("Unplugging dvd"));
	}

}

void QemuVmController::start() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		domain.start();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Starting vm"));
	}
}

void QemuVmController::stop() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		domain.stop();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Stopping vm"));
	}
}

void QemuVmController::type(const std::string& text) {
	try {
		for (auto c: text) {
			auto buttons = charmap.find(c);
			if (buttons == charmap.end()) {
				throw std::runtime_error("Unknown character to type");
			}

			press(buttons->second);
			std::this_thread::sleep_for(std::chrono::milliseconds(30));
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Typing {}", text)));
	}
}

bool QemuVmController::wait(const std::string& text, const nlohmann::json& params, const std::string& time) {

	auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(time_to_seconds(time));

	while (std::chrono::system_clock::now() < deadline) {
		if (check(text, params)) {
			return true;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	return false;
}

bool QemuVmController::check(const std::string& text, const nlohmann::json& params) {
	auto domain = qemu_connect.domain_lookup_by_name(name());
	auto stream = qemu_connect.new_stream();
	auto mime = domain.screenshot(stream);

	size_t bytes = stream.recv_all(screenshot_buffer.data(), screenshot_buffer.size());

	stream.finish();

	stb::Image screenshot(screenshot_buffer.data(), bytes);

	return shit.stink_even_stronger(screenshot, text);
}

int QemuVmController::run(const fs::path& exe, std::vector<std::string> args) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		Negotiator helper(domain);

		std::string command = exe.generic_string();
		for (auto& arg: args) {
			command += " ";
			command += arg;
		}

		return helper.execute(command);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Run guest process"));
	}
}

bool QemuVmController::has_snapshot(const std::string& snapshot) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		auto snapshots = domain.snapshots();
		for (auto& snap: snapshots) {
			if (snap.name() == snapshot) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking whether vm has snapshot {}", snapshot)));
	}
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
		std::throw_with_nested(std::runtime_error("Checking whether vm is running"));
	}
}

bool QemuVmController::is_additions_installed() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(name());
		Negotiator helper(domain);
		return helper.is_avaliable();
	} catch (const std::exception& error) {
		return false;
	}
}


void QemuVmController::copy_to_guest(const fs::path& src, const fs::path& dst) {
	try {
		//1) if there's no src on host - fuck you
		if (!fs::exists(src)) {
			throw std::runtime_error("Source file/folder does not exist on host: " + src.generic_string());
		}

		auto domain = qemu_connect.domain_lookup_by_name(name());
		Negotiator helper(domain);

		helper.copy_to_guest(src, dst);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Copying file(s) to the guest"));
	}
}

void QemuVmController::copy_from_guest(const fs::path& src, const fs::path& dst) {
	try {


		auto domain = qemu_connect.domain_lookup_by_name(name());
		Negotiator helper(domain);

		helper.copy_from_guest(src, dst);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Copying file(s) to the guest"));
	}
}

void QemuVmController::remove_from_guest(const fs::path& obj) {
	//TODO!!
}

void QemuVmController::prepare_networks() {
	try {
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
					std::string string_config = fmt::format(R"(
						<network>
							<name>{}</name>
							<bridge name="{}"/>
					)", network_to_lookup, network_to_lookup);

					if (network_to_lookup == "testo-nat") {
						string_config += fmt::format(R"(
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

					string_config += "\n</network>";
					pugi::xml_document xml_config;
					xml_config.load_string(string_config.c_str());
					auto network = qemu_connect.network_define_xml(xml_config);
					network.start();
				}
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Preparing netowkrs"));
	}

}

void QemuVmController::remove_disk() {
	try {
		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");

		auto vol_name = name() + ".img";

		for (auto& vol: pool.volumes()) {
			if (vol.name() == vol_name) {
				vol.erase();
				break;
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Removing existing disks"));
	}

}

void QemuVmController::create_disk() {
	try {
		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");
		pugi::xml_document xml_config;
		xml_config.load_string(fmt::format(R"(
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
		)", name(), config.at("disk_size").get<uint32_t>(), pool.path().generic_string(), name()).c_str());

		auto volume = pool.volume_create_xml(xml_config, {VIR_STORAGE_VOL_CREATE_PREALLOC_METADATA});
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Creating disks"));
	}

}

void QemuVmController::delete_snapshot_with_children(vir::Snapshot& snapshot) {
	try {
		auto children = snapshot.children();

		for (auto& snap: children) {
			delete_snapshot_with_children(snap);
		}
		snapshot.destroy();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Deleting snapshot with children"));
	}

}

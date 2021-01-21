
#include "QemuVM.hpp"
#include "QemuFlashDrive.hpp"
#include "QemuGuestAdditions.hpp"
#include "QemuEnvironment.hpp"
#include <base64.hpp>

#include <fmt/format.h>
#include <thread>

QemuVM::QemuVM(const nlohmann::json& config_): VM(config_),
	qemu_connect(vir::connect_open("qemu:///system"))
{
	disk_targets = {
		"hda",
		"hdb",
		"hdc",
		"hdd"
	};

	if (config.count("disk")) {
		auto disks = config.at("disk");

		if (disks.size() > disk_targets.size() - 1) {
			throw std::runtime_error("Constructing VM \"" + id() + "\" error: too many disks specified, maximum amount: " + std::to_string(disk_targets.size() - 1));
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
					throw std::runtime_error("Constructing VM \"" + id() + "\" error: nic \"" +
						nic.at("name").get<std::string>() + "\" has unsupported adapter type: \"" + driver + "\"");
				}
			}
		}
	}

	if (config.count("video")) {
		auto videos = config.at("video");

		for (auto& video: videos) {
			auto video_model = video.value("qemu_mode", preferable_video_model());

			if ((video_model != "vmvga") &&
				(video_model != "vga") &&
				(video_model != "xen") &&
				(video_model != "virtio") &&
				(video_model != "qxl") &&
				(video_model != "cirrus"))
			{
				throw std::runtime_error("Constructing VM \"" + id() + "\" error: unsupported qemu_mode \"" + video_model + "\" for video " + video.at("name").get<std::string>());
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
		{"F1", 59},
		{"F2", 60},
		{"F3", 61},
		{"F4", 62},
		{"F5", 63},
		{"F6", 64},
		{"F7", 65},
		{"F8", 66},
		{"F9", 67},
		{"F10", 68},
		{"F11", 87},
		{"F12", 88},
		{"NUMLOCK", 69},
		{"KP_0", 82},
		{"KP_1", 79},
		{"KP_2", 80},
		{"KP_3", 81},
		{"KP_4", 75},
		{"KP_5", 76},
		{"KP_6", 77},
		{"KP_7", 71},
		{"KP_8", 72},
		{"KP_9", 73},
		{"KP_PLUS", 78},
		{"KP_MINUS", 74},
		{"KP_SLASH", 98},
		{"KP_ASTERISK", 55},
		{"KP_ENTER", 96},
		{"KP_DOT", 83},
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
		{"LEFTMETA", 125},
		{"RIGHTMETA", 126},
		{"SCROLLUP", 177},
		{"SCROLLDOWN", 178},
	});
}

QemuVM::~QemuVM() {
	if (!is_defined()) {
		remove_disks();
	}
}

void QemuVM::install() {
	try {
		//now create disks
		create_disks();

		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");

		std::string string_config = fmt::format(R"(
			<domain type='kvm'>
				<name>{}</name>
				<memory unit='MiB'>{}</memory>
				<vcpu placement='static'>{}</vcpu>
				<resource>
					<partition>/machine</partition>
				</resource>
				<os>
					<type>hvm</type>
					<boot dev='cdrom'/>
					<boot dev='hd'/>
					<bootmenu enable='yes' timeout='1000'/>
				</os>
				<features>
					<acpi/>
					<apic/>
					<vmport state='off'/>
				</features>
				<cpu mode='host-passthrough'>
					<model fallback='forbid'/>
					<topology sockets='1' cores='{}' threads='1'/>
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
				<metadata>
					<testo:is_testo_related xmlns:testo='http://testo' value='true'/>
				</metadata>
		)", id(), config.at("ram").get<uint32_t>(), config.at("cpus").get<uint32_t>(), config.at("cpus").get<uint32_t>());

		string_config += R"(
			<os>
				<type>hvm</type>
				<boot dev='cdrom'/>
				<boot dev='hd'/>
		)";

		if (config.count("loader")) {
			string_config += fmt::format(R"(
				<loader readonly='yes' type='rom'>{}</loader>
			)", config.at("loader").get<std::string>());
		}

		string_config += R"(
			</os>
		)";

		string_config += R"(
			<devices>
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
				<redirdev bus='usb' type='spicevmc'>
				</redirdev>
				<redirdev bus='usb' type='spicevmc'>
				</redirdev>
				<memballoon model='virtio'>
				</memballoon>
		)";

		if (!config.count("qemu_enable_usb3")) {
			config["qemu_enable_usb3"] = true;
		}

		if (config.at("qemu_enable_usb3")) {
			string_config += R"(
				<controller type='usb' index='0' model='nec-xhci' ports='15'>
				</controller>
			)";
		} else {
			string_config += R"(
				<controller type='usb' index='0' model='ich9-ehci1'>
				</controller>
				<controller type='usb' index='0' model='ich9-uhci1'>
				</controller>
				<controller type='usb' index='0' model='ich9-uhci2'>
				</controller>
				<controller type='usb' index='0' model='ich9-uhci3'>
				</controller>
			)";
		}

		string_config += R"(
			<video>
		)";

		if (config.count("video")) {
			auto videos = config.at("video");
			for (auto& video: videos) {
				auto video_model = video.value("qemu_mode", preferable_video_model());

				string_config += fmt::format(R"(
					<model type='{}' heads='1' primary='yes'/>
				)", video_model);
			}
		} else {
			string_config += fmt::format(R"(
				<model type='{}' heads='1' primary='yes'/>
			)", preferable_video_model());
		}

		string_config += R"(
			</video>
		)";

		size_t i = 0;

		if (config.count("disk")) {
			auto disks = config.at("disk");
			for (i = 0; i < disks.size(); i++) {
				auto& disk = disks[i];
				fs::path volume_path = pool.path() / (id() + "@" + disk.at("name").get<std::string>() + ".img");
				string_config += fmt::format(R"(
					<disk type='file' device='disk'>
						<driver name='qemu' type='qcow2'/>
						<source file='{}'/>
						<target dev='{}' bus='ide'/>
						<alias name='ua-{}'/>
					</disk>
				)", volume_path.generic_string(), disk_targets[i], disk.at("name").get<std::string>());
			}
		}

		if (config.count("iso")) {
			string_config += fmt::format(R"(
				<disk type='file' device='cdrom'>
					<driver name='qemu' type='raw'/>
					<source file='{}'/>
					<target dev='{}' bus='ide'/>
					<readonly/>
				</disk>
			)", config.at("iso").get<std::string>(), disk_targets[i]);
		} else {
			string_config += fmt::format(R"(
				<disk type='file' device='cdrom'>
					<driver name='qemu' type='raw'/>
					<target dev='{}' bus='ide'/>
					<readonly/>
				</disk>
			)", disk_targets[i]);
		}

		if (!config.count("qemu_spice_agent")) {
			config["qemu_spice_agent"] = false;
		}

		if (config.at("qemu_spice_agent")) {
			string_config += R"(
			<channel type='spicevmc'>
				<target type='virtio' name='com.redhat.spice.0'/>
			</channel>)";
		}

		string_config += "\n </devices> \n </domain>";

		pugi::xml_document xml_config;
		xml_config.load_string(string_config.c_str());
		qemu_connect.domain_define_xml(xml_config);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Performing install")));
	}
}

void QemuVM::undefine() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.undefine();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Undefining vm {}", id())));
	}
}

nlohmann::json QemuVM::make_snapshot(const std::string& snapshot) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		pugi::xml_document xml_config;
		xml_config.load_string(fmt::format(R"(
			<domainsnapshot>
				<name>{}</name>
			</domainsnapshot>
			)", snapshot).c_str());

		auto snap = domain.snapshot_create_xml(xml_config);

		// If we created the _init snapshot
		if (domain.snapshots().size() == 1) {
			snap.destroy();
			snap = domain.snapshot_create_xml(xml_config);
		}

		auto new_config = domain.dump_xml();
		std::stringstream ss;
		new_config.save(ss,"  ");
		auto result = nlohmann::json::object();
		result["config"] = base64_encode((uint8_t*)ss.str().c_str(), ss.str().length());
		return result;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Taking snapshot {}", snapshot)));
	}

}

void QemuVM::rollback(const std::string& snapshot, const nlohmann::json& opaque) {
	try {
		auto config_str = opaque["config"].get<std::string>();
		auto config = base64_decode(config_str);
		std::stringstream ss;
		ss.write((const char*)&config[0], config.size());

		pugi::xml_document config_xml;
		config_xml.load_string(ss.str().c_str());

		auto domain = qemu_connect.domain_define_xml(config_xml);
		auto snap = domain.snapshot_lookup_by_name(snapshot);

		if (domain.state() != VIR_DOMAIN_SHUTOFF) {
			domain.stop();
		}
		domain.revert_to_snapshot(snap);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Performing rollback error"));
	}
}

void QemuVM::press(const std::vector<std::string>& buttons) {
	try {
		std::vector<uint32_t> keycodes;
		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);
			keycodes.push_back(scancodes[button]);
		}
		qemu_connect.domain_lookup_by_name(id()).send_keys(VIR_KEYCODE_SET_LINUX, 0, keycodes);
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Pressing buttons error"));
	}
}

void QemuVM::hold(const std::vector<std::string>& buttons) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		nlohmann::json json_command({
			{"execute", "input-send-event"},
			{"arguments", {
				{"events", nlohmann::json::array()}
			}}
		});

		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);

			uint32_t scancode = scancodes[button];
			nlohmann::json button_spec = nlohmann::json::parse(fmt::format(R"(
				{{
					"type": "key",
					"data": {{
						"down": true,
						"key": {{
							"type": "number",
							"data": {}
						}}
					}}
				}}
			)", scancode));

			json_command["arguments"]["events"].push_back(button_spec);
		}

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Holding buttons error"));
	}
}


void QemuVM::release(const std::vector<std::string>& buttons) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		nlohmann::json json_command({
			{"execute", "input-send-event"},
			{"arguments", {
				{"events", nlohmann::json::array()}
			}}
		});

		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);

			uint32_t scancode = scancodes[button];
			nlohmann::json button_spec = nlohmann::json::parse(fmt::format(R"(
				{{
					"type": "key",
					"data": {{
						"down": false,
						"key": {{
							"type": "number",
							"data": {}
						}}
					}}
				}}
			)", scancode));

			json_command["arguments"]["events"].push_back(button_spec);
		}

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Releasing buttons error"));
	}
}

void QemuVM::mouse_move_abs(uint32_t x, uint32_t y) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto tmp_screen = screenshot();

		double x_pos = double(32768) / double(tmp_screen.w) * double(x);
		double y_pos = double(32768) / double(tmp_screen.h) * double(y);

		if ((int)x_pos == 0) {
			x_pos = 1;
		}

		if ((int)y_pos == 0) {
			y_pos = 1;
		}

		nlohmann::json json_command = nlohmann::json::parse(fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "abs",
							"data": {{
								"axis": "x",
								"value": {}
							}}

						}},
						{{
							"type": "abs",
							"data": {{
								"axis": "y",
								"value": {}
							}}
						}}
					]
				}}
			}}
		)", (int)x_pos, (int)y_pos));

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse move error"));
	}
}

void QemuVM::mouse_move_rel(int x, int y) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		nlohmann::json json_command = nlohmann::json::parse(fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "rel",
							"data": {{
								"axis": "x",
								"value": {}
							}}
						}},
						{{
							"type": "rel",
							"data": {{
								"axis": "y",
								"value": {}
							}}
						}}
					]
				}}
			}}
		)", x, y));

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse move error"));
	}
}

void QemuVM::mouse_hold(const std::vector<MouseButton>& buttons) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		nlohmann::json json_command = R"(
			{
				"execute": "input-send-event",
				"arguments": {
					"events": [
					]
				}
			}
		)"_json;

		for (auto& button: buttons) {
			json_command.at("arguments").at("events").push_back({
				{"type", "btn"},
				{"data", {
					{"down", true},
					{"button", mouse_button_to_str(button)}
				}}
			});
		}

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse press buttons error"));
	}
}


void QemuVM::mouse_release(const std::vector<MouseButton>& buttons) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		nlohmann::json json_command = R"(
			{
				"execute": "input-send-event",
				"arguments": {
					"events": [
					]
				}
			}
		)"_json;

		for (auto& button: buttons) {
			json_command.at("arguments").at("events").push_back({
				{"type", "btn"},
				{"data", {
					{"down", false},
					{"button", mouse_button_to_str(button)}
				}}
			});
		}

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse release buttons error"));
	}
}

bool QemuVM::is_nic_plugged(const std::string& pci_addr) const {
	try {
		auto config = qemu_connect.domain_lookup_by_name(id()).dump_xml();
		auto devices = config.first_child().child("devices");

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			std::string pci_address;
			pci_address += std::string(nic_node.child("address").attribute("bus").value()).substr(2) + ":";
			pci_address += std::string(nic_node.child("address").attribute("slot").value()).substr(2) + ".";
			pci_address += std::string(nic_node.child("address").attribute("function").value()).substr(2);

			if (pci_address == pci_addr) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking if nic {} is plugged", pci_addr)));
	}
}

std::set<std::string> QemuVM::plugged_nics() const {
	auto domain = qemu_connect.domain_lookup_by_name(id());
	auto xml_config = domain.dump_xml();

	auto devices = xml_config.first_child().child("devices");

	std::set<std::string> result;

	for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
		if (std::string(nic_node.attribute("type").value()) != "network") {
			continue;
		}

		std::string pci_address;
		pci_address += std::string(nic_node.child("address").attribute("bus").value()).substr(2) + ":";
		pci_address += std::string(nic_node.child("address").attribute("slot").value()).substr(2) + ".";
		pci_address += std::string(nic_node.child("address").attribute("function").value()).substr(2);

		result.insert(pci_address);
	}

	return result;
}

std::string QemuVM::attach_nic(const std::string& nic) {
	try {
		std::string string_config;

		for (auto& nic_json: config.at("nic")) {
			if (nic_json.at("name") == nic) {
				std::string source_network = config.at("prefix").get<std::string>();
				source_network += nic_json.at("attached_to").get<std::string>();

				string_config += fmt::format(R"(
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
		auto domain = qemu_connect.domain_lookup_by_name(id());

		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document nic_config;
		nic_config.load_string(string_config.c_str());

		auto already_plugged_nics = plugged_nics();
		domain.attach_device(nic_config, flags);

		auto new_plugged_nics = plugged_nics();

		std::set<std::string> diff;

		std::set_difference(new_plugged_nics.begin(), new_plugged_nics.end(), already_plugged_nics.begin(), already_plugged_nics.end(),
			std::inserter(diff, diff.begin()));

		return *diff.begin();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Attaching nic {}", nic)));
	}
}

void QemuVM::detach_nic(const std::string& pci_addr) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			std::string pci_address;
			pci_address += std::string(nic_node.child("address").attribute("bus").value()).substr(2) + ":";
			pci_address += std::string(nic_node.child("address").attribute("slot").value()).substr(2) + ".";
			pci_address += std::string(nic_node.child("address").attribute("function").value()).substr(2);

			if (pci_address == pci_addr) {
				domain.detach_device(nic_node, flags);
				return;
			}
		}

		throw std::runtime_error("Nic with address " + pci_addr + " not found");
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Detaching nic {}", pci_addr)));
	}
}

bool QemuVM::is_link_plugged(const std::string& pci_addr) const {
	try {
		auto config = qemu_connect.domain_lookup_by_name(id()).dump_xml();
		auto devices = config.first_child().child("devices");
		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			std::string pci_address;
			pci_address += std::string(nic_node.child("address").attribute("bus").value()).substr(2) + ":";
			pci_address += std::string(nic_node.child("address").attribute("slot").value()).substr(2) + ".";
			pci_address += std::string(nic_node.child("address").attribute("function").value()).substr(2);

			if (pci_address == pci_addr) {
				if (nic_node.child("link").empty()) {
					return false;
				}

				std::string state = nic_node.child("link").attribute("state").value();

				std::cout << "The state is: " << state << std::endl;
				if (state == "up") {
					return true;
				} else if (state == "down") {
					return false;
				}
			}
		}
		throw std::runtime_error("Nic with address " + pci_addr + " not found");
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking link status on nic {}", pci_addr)));
	}
}

void QemuVM::set_link(const std::string& pci_addr, bool is_connected) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");
		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			std::string pci_address;
			pci_address += std::string(nic_node.child("address").attribute("bus").value()).substr(2) + ":";
			pci_address += std::string(nic_node.child("address").attribute("slot").value()).substr(2) + ".";
			pci_address += std::string(nic_node.child("address").attribute("function").value()).substr(2);

			if (pci_address == pci_addr) {
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

				std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

				if (domain.is_active()) {
					flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
				}

				domain.update_device(nic_node, flags);
				return;
			}
		}
		throw std::runtime_error("Nic with address " + pci_addr + " not found");
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting link status on nic {}", pci_addr)));
	}
}

bool QemuVM::is_flash_plugged(std::shared_ptr<FlashDrive> fd) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		std::string result = "";

		for (auto disk = devices.child("disk"); disk; disk = disk.next_sibling("disk")) {
			if (std::string(disk.attribute("device").value()) != "disk") {
				continue;
			}

			if (std::string(disk.child("target").attribute("dev").value()) == "sdb") {
				result = disk.child("source").attribute("file").value();

				//nullptr fd means "Any" flash drive
				if (!fd) {
					return true;
				}

				if (result == fd->img_path().generic_string()) {
					return true;
				}
			}
		}

		return false;
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking if flash drive {} is plugged", fd->name())));
	}
}

void QemuVM::attach_flash_drive(const std::string& img_path) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		std::string string_config = fmt::format(R"(
			<disk type='file'>
				<driver name='qemu' type='qcow2'/>
				<source file='{}'/>
				<target dev='sdb' bus='usb' removable='on'/>
			</disk>
			)", img_path);

		//we just need to create new device
		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_CURRENT};

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

void QemuVM::plug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	try {
		attach_flash_drive(fd->img_path());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Plugging flash drive {}", fd->name())));
	}
}

bool QemuVM::is_hostdev_plugged() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		for (auto hostdev = devices.child("hostdev"); hostdev; hostdev = hostdev.next_sibling("hostdev")) {
			return true;
		}

		return false;

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking for plugged hostdevs")));
	}
}

void QemuVM::plug_hostdev_usb(const std::string& addr) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		auto parsed_addr = parse_usb_addr(addr);

		std::string string_config = fmt::format(R"(
			<hostdev mode='subsystem' type='usb'>
				<source>
					<address bus='{:#x}' device='{:#x}'/>
				</source>
			  </hostdev>
			)", parsed_addr.first, parsed_addr.second);

		//we just need to create new device
		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_CURRENT};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document disk_config;
		disk_config.load_string(string_config.c_str());

		domain.attach_device(disk_config, flags);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Plugging host dev usb device {}", addr)));
	}
}

void QemuVM::unplug_hostdev_usb(const std::string& addr) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		auto parsed_addr = parse_usb_addr(addr);

		bool found = false;

		for (auto hostdev = devices.child("hostdev"); hostdev; hostdev = hostdev.next_sibling("hostdev")) {
			auto hostdev_addr = hostdev.child("source").child("address");

			int bus_id = std::stoi(hostdev_addr.attribute("bus").value(), 0, 0);
			int dev_id = std::stoi(hostdev_addr.attribute("device").value(), 0, 0);

			if ((bus_id == parsed_addr.first)
				&& (dev_id == parsed_addr.second))
			{
				domain.detach_device(hostdev, flags);
				found = true;
				break;
			}
		}

		if (!found) {
			throw std::runtime_error("Requested usb device is not plugged into the virtual machine");
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Unplugging host dev usb device {}", addr)));
	}
}

void QemuVM::detach_flash_drive() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		for (auto disk = devices.child("disk"); disk; disk = disk.next_sibling("disk")) {
			if (std::string(disk.attribute("device").value()) != "disk") {
				continue;
			}

			if (std::string(disk.child("target").attribute("dev").value()) == "sdb") {
				domain.detach_device(disk, flags);
				break;
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Detaching flash drive"));
	}
}

//for now it's just only one flash drive possible
void QemuVM::unplug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	try {
		detach_flash_drive();
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Unplugging flash drive {}", fd->name())));
	}
}


bool QemuVM::is_dvd_plugged() const {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");
		if (cdrom.child("source").empty()) {
			return false;
		}
		return !cdrom.child("source").attribute("file").empty();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Checking if dvd is plugged"));
	}
}

void QemuVM::plug_dvd(fs::path path) {
	try {
		if (!fs::exists(path)) {
			throw std::runtime_error(std::string("specified iso file does not exist: ")
				+ path.generic_string());
		}

		if (!fs::is_regular_file(path)) {
			throw std::runtime_error(std::string("specified iso is not a regular file: ")
				+ path.generic_string());
		}
		auto domain = qemu_connect.domain_lookup_by_name(id());

		if (is_dvd_plugged()) {
			throw std::runtime_error("Some dvd is already plugged in");
		}

		std::string string_config = fmt::format(R"(
			<disk type='file' device='cdrom'>
				<driver name='qemu' type='raw'/>
				<source file='{}'/>
				<backingStore/>
				<target dev='hdb' bus='ide'/>
				<readonly/>
				<alias name='ide0-0-1'/>
				<address type='drive' controller='0' bus='0' target='0' unit='1'/>
			</disk>
		)", path.generic_string().c_str());

		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_FORCE};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document dvd_config;
		dvd_config.load_string(string_config.c_str());

		domain.update_device(dvd_config, flags);
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("plugging dvd {}", path.generic_string())));
	}
}

void QemuVM::unplug_dvd() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");

		if (!is_dvd_plugged()) {
			throw std::runtime_error("Dvd is already unplugged");
		}

		cdrom.remove_child("source");

		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_FORCE};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		domain.update_device(cdrom, flags);
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error("Unplugging dvd"));
	}

}

void QemuVM::start() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto xml = domain.dump_xml();
		xml.first_child().child("cpu");
		pugi::xml_document cpu;
		cpu.load_string(fmt::format(R"(
			<cpu mode='host-passthrough'>
				<model fallback='forbid'/>
				<topology sockets='1' cores='{}' threads='1'/>
			</cpu>
		)", config.at("cpus").get<uint32_t>()).c_str());
		xml.first_child().append_copy(cpu.first_child());
		qemu_connect.domain_define_xml(xml);
		domain = qemu_connect.domain_lookup_by_name(id());
		domain.start();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Starting vm"));
	}
}

void QemuVM::stop() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.stop();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Stopping vm"));
	}
}

void QemuVM::power_button() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.shutdown();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Shutdowning vm"));
	}
}

void QemuVM::suspend() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.suspend();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Suspending vm"));
	}
}

void QemuVM::resume() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.resume();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Resuming vm"));
	}
}

stb::Image<stb::RGB> QemuVM::screenshot() {
	auto domain = qemu_connect.domain_lookup_by_name(id());
	auto stream = qemu_connect.new_stream();
	auto mime = domain.screenshot(stream);

	if (!screenshot_buffer.size()) {
		screenshot_buffer.resize(10'000'000);
	}

	size_t bytes = stream.recv_all(screenshot_buffer.data(), screenshot_buffer.size());

	stream.finish();

	stb::Image<stb::RGB> screenshot(screenshot_buffer.data(), bytes);
	return screenshot;
}

int QemuVM::run(const fs::path& exe, std::vector<std::string> args,
	const std::function<void(const std::string&)>& callback) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		QemuGuestAdditions helper(domain);

		std::string command = exe.generic_string();
		for (auto& arg: args) {
			command += " ";
			command += arg;
		}

		return helper.execute(command, callback);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Run guest process"));
	}
}

bool QemuVM::has_snapshot(const std::string& snapshot) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
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

void QemuVM::delete_snapshot(const std::string& snapshot) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto vir_snapshot = domain.snapshot_lookup_by_name(snapshot);
		vir_snapshot.destroy();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Deleting snapshot with children"));
	}
}

bool QemuVM::is_defined() const {
	auto domains = qemu_connect.domains({VIR_CONNECT_LIST_DOMAINS_PERSISTENT});
	for (auto& domain: domains) {
		if (domain.name() == id()) {
			return true;
		}
	}
	return false;
}

VmState QemuVM::state() const {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto state = domain.state();
		if (state == VIR_DOMAIN_SHUTOFF) {
			return VmState::Stopped;
		} else if (state == VIR_DOMAIN_RUNNING) {
			return VmState::Running;
		} else if (state == VIR_DOMAIN_PAUSED) {
			return VmState::Suspended;
		} else {
			return VmState::Other;
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Checking whether vm is running"));
	}
}

bool QemuVM::is_additions_installed() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		QemuGuestAdditions helper(domain);
		return helper.is_avaliable();
	} catch (const std::exception& error) {
		return false;
	}
}

std::string QemuVM::get_tmp_dir() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		QemuGuestAdditions helper(domain);
		return helper.get_tmp_dir();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Getting tmp directory path on guest"));
	}
}

void QemuVM::copy_to_guest(const fs::path& src, const fs::path& dst) {
	try {
		//1) if there's no src on host - fuck you
		if (!fs::exists(src)) {
			throw std::runtime_error("Source file/folder does not exist on host: " + src.generic_string());
		}

		auto domain = qemu_connect.domain_lookup_by_name(id());
		QemuGuestAdditions helper(domain);

		helper.copy_to_guest(src, dst);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Copying file(s) to the guest"));
	}
}

void QemuVM::copy_from_guest(const fs::path& src, const fs::path& dst) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		QemuGuestAdditions helper(domain);

		helper.copy_from_guest(src, dst);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Copying file(s) from the guest"));
	}
}

void QemuVM::remove_from_guest(const fs::path& obj) {
	//TODO!!
}

void QemuVM::remove_disks() {
	try {
		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");

		//TODO

		for (auto& vol: pool.volumes()) {
			std::string volume_name = vol.name();
			if (volume_name.find("@") == std::string::npos) {
				continue;
			}
			volume_name = volume_name.substr(0, volume_name.find("@"));
			if (volume_name == id()) {
				vol.erase();
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Removing existing disks"));
	}
}

void QemuVM::create_new_disk(const std::string& name, uint32_t size) {
	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");

	fs::path disk_path = pool.path() / (name + ".img");

	pugi::xml_document xml_config;
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
	)", name, size, disk_path.generic_string()).c_str());

	auto volume = pool.volume_create_xml(xml_config, {VIR_STORAGE_VOL_CREATE_PREALLOC_METADATA});
}

void QemuVM::import_disk(const std::string& name, const fs::path& source) {
	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");
	fs::path disk_path = pool.path() / (name + ".img");
	fs_copy(source, disk_path);

	pool.refresh();
}

void QemuVM::create_disks() {
	try {
		if (!config.count("disk")) {
			return;
		}
		auto disks = config.at("disk");
		for (size_t i = 0; i < disks.size(); ++i) {
			auto& disk = disks[i];
			std::string disk_name = id() + "@" + disk.at("name").get<std::string>();

			if (disk.count("source")) {
				fs::path source_disk = disk.at("source").get<std::string>();
				import_disk(disk_name, source_disk);
			} else {
				create_new_disk(disk_name, disk.at("size").get<uint32_t>());
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Creating disks"));
	}
}

std::string QemuVM::preferable_video_model() {
	auto dom_caps = qemu_connect.get_domain_capabilities();
	auto models_node = dom_caps.first_child().child("devices").child("video").child("enum");

	std::set<std::string> models;
	std::vector<std::string> preferable = {
		"vmvga",
		"qxl",
		"cirrus"
	};

	for (auto model = models_node.child("value"); model; model = model.next_sibling("value")) {
		models.insert(model.text().as_string());
	}

	for (auto& model: preferable) {
		if (models.find(model) != models.end()) {
			return model;
		}
	}

	throw std::runtime_error("Can't find any acceptable video model");
}

std::string QemuVM::mouse_button_to_str(MouseButton btn) {
	switch (btn) {
		case Left: return "left";
		case Right: return "right";
		case Middle: return "middle";
		case WheelUp: return "wheel-up";
		case WheelDown: return "wheel-down";
		default: throw std::runtime_error("Unknown button: " + btn);
	}
}

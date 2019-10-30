
#include "QemuNetwork.hpp"

#include "pugixml/pugixml.hpp"
#include <fmt/format.h>

QemuNetwork::QemuNetwork(const nlohmann::json& config): Network(config), qemu_connect(vir::connect_open("qemu:///system"))
{
	if (!is_defined()) {
		return;
	}

	auto network = qemu_connect.network_lookup_by_name(id());

	if (!network.is_active()) {
		network.start();
	}
}

bool QemuNetwork::is_defined() {
	for (auto& network: qemu_connect.networks()) {
		if (network.name() == id()) {
			return true;
		}
	}
	return false;
}

void QemuNetwork::create() {
	try {
		remove_if_exists();

		std::string string_config = fmt::format(R"(
			<network>
				<name>{}</name>
				<bridge name="{}"/>
		)", id(), id());

		auto mode = config.at("mode").get<std::string>();

		if (mode == "nat") {
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

		bool autostart = config.value("autostart", true);

		network.set_autostart(autostart);
		network.start();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Creating network"));
	}
}

void QemuNetwork::remove_if_exists() {
	for (auto& network: qemu_connect.networks()) {
		if (network.name() == id()) {
			if (network.is_active()) {
				network.stop();
			}
			if (network.is_persistent()) {
				network.undefine();
			}
		}
	}
}


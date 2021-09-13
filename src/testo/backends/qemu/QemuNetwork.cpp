
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


std::string QemuNetwork::find_free_nat() const {
	for (int i = 179; i < 254; i++) {
		std::string network_to_look("192.168.");
		network_to_look += std::to_string(i);
		network_to_look += ".1";
		auto is_free = true;
		for (auto& network: qemu_connect.networks()) {
			auto config = network.dump_xml();
			auto forward = config.first_child().child("forward");
			if (!forward) {
				continue;
			}
			if (std::string(forward.attribute("mode").value()) != "nat") {
				continue;
			}

			auto ip = config.first_child().child("ip");
			if (!ip) {
				continue;
			}

			if (std::string(ip.attribute("address").value()) == network_to_look) {
				is_free = false;
				break;
			}
		}
		if (is_free) {
			return std::to_string(i);
		}
	}
	throw std::runtime_error("Can't find a free nat to create network " + id());
}

void QemuNetwork::create() {
	try {
		if (is_defined()) {
			undefine();
		}

		std::string string_config = fmt::format(R"(
			<network>
				<name>{}</name>
				<bridge name="{}"/>
		)", id(), id());

		auto mode = config.at("mode").get<std::string>();

		if (mode == "nat") {
			auto network = find_free_nat();
			string_config += fmt::format(R"(
				<forward mode='nat'>
					<nat>
						<port start='1024' end='65535'/>
					</nat>
				</forward>
				<ip address='192.168.{}.1' netmask='255.255.255.0'>
					<dhcp>
						<range start='192.168.{}.2' end='192.168.{}.254'/>
					</dhcp>
				</ip>
			)", network, network, network);
		}

		string_config += "\n</network>";
		pugi::xml_document xml_config;
		xml_config.load_string(string_config.c_str());
		auto network = qemu_connect.network_define_xml(xml_config);

		network.set_autostart(true);
		network.start();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Creating network"));
	}
}

void QemuNetwork::undefine() {
	try {
		auto network = qemu_connect.network_lookup_by_name(id());
		if (network.is_active()) {
			network.stop();
		}
		if (network.is_persistent()) {
			network.undefine();
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Deleting network"));
	}
}


#include "QemuNetwork.hpp"

QemuNetwork::QemuNetwork(const nlohmann::json& config): Network(config), qemu_connect(vir::connect_open("qemu:///system"))
{

}

bool QemuNetwork::is_defined() {
	for (auto& network: qemu_connect.networks()) {
		if (network.name() == id()) {
			return true;
		}
	}
	return false;
}
